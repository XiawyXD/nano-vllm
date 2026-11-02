# 历史归档索引

外部归档：`/home/hrxu/nano-vllm-archive/layout-20261025-111959`。

- source-before.tar.gz：整理前完整文件快照，含未跟踪源码和既有修改；排除 Git 对象、依赖主环境、原始 artifacts 和字节码，独立 CUDA 学习目录也保留在快照中。
- source-before.json：逐文件 SHA256 与大小。
- git-before.json / working-tree-before.patch / index-before.patch：原 HEAD/index 与修改状态。
- historical/：旧阶段报告、一次性诊断、旧驱动及过时 fixture；原相对目录保留。
- evidence/artifacts/：全部原始实验产物，整体移动；仓库 artifacts 符号链接保持旧路径可用。
- plan.json / applied.json：逐项原因、移动路径和运行时未改证明。
- cpu-before.json / cpu-after-*：整理前后回归；失败尝试也保留。

本仓库 [path_map.json](path_map.json) 可按旧名称查询新位置。删除的只有可生成副本与一份完全重复文档，删除前已经进入快照。历史源码与快照不自动绑定当前项目的新目录；复现正式结果优先使用不变的 `/home/hrxu/nano-full-opt-20261024`。

不要直接把旧 tar 覆盖到当前工作树。需要恢复时解压到新的空目录再对比；这样不会丢掉整理后的修改。

主目录旧实验清理：`/home/hrxu/nano-vllm-archive/home-cleanup-20261025-120138`；见 directory-map.json 与 receipt.json。当前正式权重和校准环境保持原路径；历史源码副本压缩归档。


## 2026-11-02 GitHub 源码整理

服务器备份 `/home/hrxu/nano-vllm-archive/github-prep-20261102-001` 保留整理前6505个源码文件、未跟踪修改、Git diff/index、完整history.bundle和依赖清单。候选源树不含独立cuda-learning、模型、环境或原始artifacts。原历史中两份原始实验tar仍在完整bundle里；本地发布分支从上游bb823b3承接当前源码，不带这些实验历史对象。

报告移动与文档引用更新记录在 `layout-plan.json`，前后运行时代码保持逐文件SHA一致。GitHub上传步骤见 [指南](../docs/GITHUB_UPLOAD.md)。

原服务器是浅克隆：`git-shallow-before.txt` 与 `git-metadata-before.tar.gz` 同时保存原边界、refs、index和对象；history.bundle保留所有当时可见引用，但单独clone该bundle不能补齐上游父提交。本地发布分支从上游HTTPS获取完整的bb823b3祖先历史，再承接当前源码。
