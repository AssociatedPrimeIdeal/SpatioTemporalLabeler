SpatioTemporal Labeler 独立发行版
=================================

绿色运行
--------
Linux: 进入 SpatioTemporalLabeler 目录，运行 ./SpatioTemporalLabeler
Windows: 进入 SpatioTemporalLabeler 目录，双击 SpatioTemporalLabeler.exe

整个目录可以移动或复制。程序已包含 Python、PySide6、VTK、NumPy、pyqtgraph 和 pynrrd，目标机器不需要安装 Conda 或 Python。不要只复制单个可执行文件，依赖库位于同一目录中。

主要操作
--------
左键拖动: 使用当前画笔、橡皮擦或闭合线工具
右键拖动: 临时擦除，不切换当前工具
Shift+点击/拖动: 联动定位 X/Y/Z 切片
Shift+滚轮: 调整画笔或橡皮擦直径
空间视图滚轮: 切换正交切片
闭合线绘制后双击: 确认填充；Esc: 取消

当前用户安装
------------
Linux: 在解压目录运行 ./install.sh
Windows: 右键 install.ps1，选择“使用 PowerShell 运行”；不需要管理员权限。

卸载
----
Linux: 运行 ./uninstall.sh
Windows: 运行 uninstall.ps1

系统要求
--------
Linux x86_64 版本需要兼容的 glibc、X11/Wayland 桌面和 OpenGL 驱动。GPU 驱动属于操作系统组件，不包含在绿色包内。
Windows x64 版本支持 Windows 10/11，并使用目标机器的显卡驱动。

命令行加载
----------
SpatioTemporalLabeler /path/to/sequence-directory
SpatioTemporalLabeler --image image.seq.nrrd --mask seg.seq.nrrd
