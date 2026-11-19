# 测试

从仓库根目录运行全部单元测试（无需模型权重或 GPU）：

```bash
python -m unittest discover -v
```

运行单个测试模块：

```bash
python -m unittest tests.test_custom_metrics -v
```

`test_9b_following.py` 是需要模型的诊断脚本，不会在单元测试发现过程中启动推理。单独运行：

```bash
python -m tests.test_9b_following --help
```
