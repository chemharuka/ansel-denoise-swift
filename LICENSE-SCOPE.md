# 许可范围（为什么 Photons 可以调用本仓库）

本仓库：**GPL-3.0-or-later**（`LICENSE`）。Photons：**MIT**。
两者的共存方式是 **两个独立进程**，不是链接——与同目录的 `hdrmerge-swift` 完全同构。

## 上游与衍生关系

网络定义、权重格式与 σ 校准约定来自
[`aurelienpierreeng/ansel-denoise`](https://github.com/aurelienpierreeng/ansel-denoise)（GPL-3.0），
Ansel 的 `rawdenoiseai` 模块的训练管线。本仓库是它的 **Swift + Metal 重写**，因此同样是 GPL-3.0。

具体对应关系：

| 上游（Python） | 本仓库（Swift） |
|---|---|
| `src/ansel_denoise/model.py`（~70 行 UNet） | `Sources/AnselDenoise/AnselDenoiseForward.swift` 的 `AnselDenoiseForward` |
| `src/ansel_denoise/noise.py` 的 `sigma_map` | `AnselDenoiseForward.makeInputPlanes` 的 σ 平面 |
| `docs/architecture.md` 的格式说明 | `AnselDenoiseModel`（`.anselnn` 解析） |
| `.anselnn` 权重 | 不随本仓库分发，按上游 manifest 的 sha256 拉取 |

推理执行器（Metal 内核、CPU 多线程卷积）是本仓库原创，但作为同一 GPL 作品的组成部分，
许可不因此改变。

## 判定依据（GPLv3 §5 末段；GNU GPL FAQ #MereAggregation）

> "A compilation of a covered work with other separate and independent works,
> which are not by their nature extensions of the covered work, and which are
> not combined with it such as to form a larger program … is called an
> 'aggregate' … Inclusion of a covered work in an aggregate does not cause this
> License to apply to the other parts of the aggregate."

> "If the modules are included in the same executable file, they are definitely
> combined in one program. If modules are designed to run linked together in a
> shared address space, that almost surely means combining them into one program.
> By contrast, pipes, sockets and command-line arguments are communication
> mechanisms normally used between two separate programs."

## 本仓库遵守的边界

1. **不被链接进 Photons。** Photons 的 `Package.swift` 不引用本仓库；本仓库是独立的 SPM 包，
   产出独立的 `ansel-denoise-swift` 可执行文件。
2. **只通过进程边界通信。** Photons 侧是 `Sources/PhotonsExtensions/Denoise/AnselDenoiseClient.swift`
   （MIT），它 `Process` 启动本程序，用命令行参数传参、stdout 收 JSON、临时文件交换 float32 像素。
   没有共享内存、没有动态库、没有 XPC。
3. **可执行文件单独随 app 分发。** 与 `hdrmerge-swift` 一样放在
   `Contents/Resources/bin/` 下，作为独立程序存在。
4. **本仓库自身保持 GPL 完整。** 源码在这里，任何人可以重建、修改、再分发。

## 上游义务

分发包含本组件的二进制包时，需按 GPL-3.0 §6 提供对应源码（本目录即是）。本目录的
`README.md` 记录了构建方式；不含上游权重（权重由用户按 manifest 自行获取）。

## 与权重的许可关系

`.anselnn` 权重由上游训练管线产出。上游对**训练数据**使用三层许可（CC0 / 逐 shard 声明的 CC /
ATDL-1.1），其中 ATDL-1.1 明确允许「训练出权重并商用」；上游**未对权重本身附加额外限制**。
本仓库不分发权重，只分发推理代码，因此不受数据许可约束。

商用前建议向上游确认一次权重许可——这是本仓库无法代为判断的部分。
