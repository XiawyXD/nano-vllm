# GitHub 上传与后续管理

## 本地仓库

整理后的独立仓库：

```text
C:\Users\mayonnaise\Documents\ai infra\nano-vllm
```

当前发布分支为 `codex/github-ready`，`origin` 指向 `https://github.com/XiawyXD/nano-vllm.git`，`upstream` 指向原 Nano-vLLM。外层 `ai infra` 也是另一个 Git 工作区；上传此项目时进入上面的 `nano-vllm`，不在外层执行 `git add .`。

本地分支从上游基线 `bb823b3` 承接当前完整源码。原服务器的开发分支、未提交修改前镜像、Git 元数据和浅克隆边界另有备份。本地已从上游获取该基线的完整祖先历史；旧实验历史中的两份原始数据压缩包不进入本发布分支。

## 第一次上传

在 PowerShell 执行：

```powershell
Set-Location 'C:\Users\mayonnaise\Documents\ai infra\nano-vllm'
git branch --show-current
git remote -v
python scripts/check_repository.py
git status --short
git add -A
git diff --cached --stat
git commit -m "Organize Nano-vLLM runtime, docs and reproducible experiments"
git push -u origin codex/github-ready
```

`python` 检查仅依赖标准库。没有 Python 命令时可以先使用 Git 上传，或换成已安装 Python 的完整路径；检查结果也已在整理归档中保存。

首次 HTTPS push 按 Git Credential Manager 的登录提示授权 GitHub；它可能打开浏览器。不要把 token 写进远程 URL、源文件或命令记录。若 Git 提示 author identity unknown，先配置你要公开在提交中的名字和邮箱，再重新 commit：

```powershell
git config user.name "你的 GitHub 名字"
git config user.email "你的 GitHub 提交邮箱或 noreply 邮箱"
```

如果目标仓库还不存在，先在 GitHub 新建 `XiawyXD/nano-vllm` 空仓库，不初始化 README/License/gitignore；如果已存在，直接推送这个新分支。上传后在仓库分支选择器查看 `codex/github-ready`，再通过 Pull Request 合并到你自己的默认分支。无需 force push。

## 日常开发

```powershell
git switch -c codex/describe-your-change
# 修改和验证
git add -A
git commit -m "Describe the concrete change"
git push -u origin HEAD
```

合并前查看 diff 和测试结果。服务器需要更新时，在确认工作区状态后 fetch/merge 对应分支；不要直接覆盖服务器尚未提交的修改。

## 服务器原仓库

服务器路径 `/home/hrxu/nano-vllm` 保留原分支与历史。其 `origin` 是上游，`mygithub` 才是个人仓库；不要把个人修改推到上游 `origin`。本轮只读检查发现服务器 GitHub SSH 主机校验尚未通过，未关闭校验或更改 SSH 配置；推荐先用上面的本地 HTTPS 路径。

如果以后使用 SSH，按 GitHub 的[连接说明](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)设置并核对主机指纹，然后先运行 `ssh -T git@github.com` 验证。

## 上传内容与归档

Git 保存源码、测试、基准、配置、文档、许可和小型 fixture；`.gitignore` 排除权重、环境、原始证据、编译扩展、源码快照和独立 CUDA 学习目录。历史实验数据仍在原服务器归档，模型另行获取。

GitHub 官方参考：[导入已有代码](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)、[管理远程仓库](https://docs.github.com/en/get-started/git-basics/managing-remote-repositories)。
