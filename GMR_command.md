# 这里列出了常用的命令行，方便直接复制使用

## 青瞳bvh格式动捕数据重定向

### 单个文件

```bash
python scripts/bvh_to_robot.py --robot ths_23dof --format qingtong --rate_limit --bvh_file motion_data/qingtong_bvh/xxx.bvh
```

### 文件夹

```bash
python scripts/bvh_to_robot_dataset.py --robot ths_23dof --format qingtong --src_folder motion_data/qingtong_bvh/walk_run --tgt_folder output/qingtong/walk_run
```