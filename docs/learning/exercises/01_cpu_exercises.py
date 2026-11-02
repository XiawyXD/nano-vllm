"""第1课的独立CPU练习模板；两个函数留给你完成，不属于项目待修代码。"""


def physical_slot(token_position, block_table, block_size=256):
    """把零基逻辑位置映射到物理slot；不用导入torch。"""
    raise NotImplementedError("请根据页号和页内偏移实现")


def minimum_kv_blocks(prompt_tokens, max_new_tokens, block_size=256):
    """无前缀复用：完成请求最多需要多少KV页？注意最后输出的KV。"""
    raise NotImplementedError("请根据真正需要进入模型的token数量实现")


if __name__ == "__main__":
    assert [physical_slot(p, [7, 2]) for p in [0, 255, 256, 257]] == [1792, 2047, 512, 513]
    assert minimum_kv_blocks(7, 2, block_size=4) == 2
    assert minimum_kv_blocks(5, 3, block_size=256) == 1
    assert minimum_kv_blocks(256, 1, block_size=256) == 1
    assert minimum_kv_blocks(256, 2, block_size=256) == 2
    print("两个CPU练习通过。继续解释每个边界为什么成立。")
