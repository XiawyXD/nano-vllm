"""逐位搬运分页 KV；只改变物理布局，不做 dtype 转换或量化。"""
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["plane_stride"])
def _gather(KV, STAGE, IDS, plane_stride, PAGE: tl.constexpr, PLANES: tl.constexpr, TILE: tl.constexpr):
    lane = tl.program_id(1)
    x = tl.program_id(0).to(tl.int64)*TILE + tl.arange(0,TILE).to(tl.int64)
    valid = x < PAGE*PLANES
    bid = tl.load(IDS+lane).to(tl.int64)
    source = (x//PAGE)*plane_stride + bid*PAGE + x%PAGE
    values = tl.load(KV+source, valid, other=0)
    tl.store(STAGE+lane*(PAGE*PLANES)+x, values, valid)


@triton.jit(do_not_specialize=["plane_stride"])
def _scatter(STAGE, KV, IDS, plane_stride, PAGE: tl.constexpr, PLANES: tl.constexpr, TILE: tl.constexpr):
    lane = tl.program_id(1)
    x = tl.program_id(0).to(tl.int64)*TILE + tl.arange(0,TILE).to(tl.int64)
    valid = x < PAGE*PLANES
    bid = tl.load(IDS+lane).to(tl.int64)
    destination = (x//PAGE)*plane_stride + bid*PAGE + x%PAGE
    values = tl.load(STAGE+lane*(PAGE*PLANES)+x, valid, other=0)
    tl.store(KV+destination, values, valid)


def move_pages(kv, stage, ids, count, *, gather):
    if not kv.is_contiguous() or kv.ndim != 6:
        raise ValueError("batched transfer requires contiguous [2,L,P,B,H,D] KV")
    page = kv.shape[3]*kv.shape[4]*kv.shape[5]
    planes = kv.shape[0]*kv.shape[1]
    grid=(triton.cdiv(page*planes,2048),count)
    if gather:
        _gather[grid](kv,stage,ids,kv.shape[2]*page,PAGE=page,PLANES=planes,TILE=2048)
    else:
        _scatter[grid](stage,kv,ids,kv.shape[2]*page,PAGE=page,PLANES=planes,TILE=2048)
