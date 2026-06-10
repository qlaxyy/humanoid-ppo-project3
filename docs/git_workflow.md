# Git and GitHub Workflow

这份文档记录本项目最常用的 Git 操作。你可以把 Git 理解成“本地版本管理”，GitHub 是“远程备份和同步平台”。

## 核心概念

工作区：你正在编辑的普通文件。

暂存区：准备放进下一次提交的改动，由 `git add` 管理。

本地仓库：你电脑上的版本历史，由 `git commit` 保存。

远程仓库：GitHub 上的仓库，由 `git push` 上传、`git pull` 下载。

## 每次改代码后的流程

查看当前状态：

```bash
git status
```

查看具体改了什么：

```bash
git diff
```

把改动加入暂存区：

```bash
git add train.py configs/ppo_humanoid_stable.json
```

保存为本地提交：

```bash
git commit -m "Add stable PPO config"
```

上传到 GitHub：

```bash
git push
```

## Colab 如何拿到 GitHub 最新代码

如果 Colab 项目目录是从 GitHub clone 出来的，可以在 Colab 里运行：

```bash
!git pull
```

如果 Colab 目录是手动上传到 Drive 的普通文件夹，没有 `.git` 目录，那么 `git pull` 不能直接用。此时有两个选择：

- 继续手动上传被修改的文件并覆盖
- 重新从 GitHub clone 一个项目目录

私密仓库 clone 需要 GitHub 身份认证。不要把 token 写进 notebook 文本里，也不要提交带 token 的 notebook。

## 不要提交的内容

这些训练产物一般放在 Google Drive，不放进 GitHub：

```text
runs/
logs/
models/
videos/
*.zip
*.pkl
*.mp4
```

GitHub 主要保存：

```text
源码
配置
说明文档
实验记录模板
报告 Markdown 草稿
```

