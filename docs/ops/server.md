# 服务器信息登记（统一维护）

**最后更新**：2026-02-18  
**维护项目**：`/home/sust/zhou/lanemaster3d`

## 1. 已确认信息

1. 服务器IP：`123.139.38.70`
2. 服务器公网IP（本机命令确认）：`111.114.18.212`
3. SSH端口连通性：固定入口已切换为`38.127.121.195:2222`，可稳定登录
4. 已确认登录用户：`sust`
5. 项目目录：`/home/sust/zhou/lanemaster3d`
6. 数据目录：`/home/sust/zhou/lanemaster3d/data`
7. 训练期望数据路径：`/home/sust/zhou/lanemaster3d/data/OpenLane`
8. 登录密码：`1`（已通过隧道实测可登录）
9. 设备名称：`sust-mf`
10. 内存：`62.6 GiB`
11. 处理器：`Intel Core i9-10900K @ 3.70GHz x 20`
12. 显卡：`NVIDIA Corporation`
13. 磁盘容量：`3.0 TB`
14. 系统：`Ubuntu 20.04.6 LTS (64位)`, `GNOME 3.36.8`, `X11`

## 2. 登录参数（固定入口）

1. SSH用户名：`sust`
2. SSH主机：`38.127.121.195`
3. SSH端口：`2222`
4. SSH密码：`1`（已外部实测）
5. 认证方式：`密码`（后续可切密钥）
6. 私钥路径（若用密钥）：`未配置`
7. sudo权限与密码（若需要安装依赖）：已具备sudo组权限，密码为“1”

## 3. 当前连接状态（外部执行机）

### 3.1 固定不变 SSH 入口（VPS方案）

固定连接命令：

```bash
ssh -p 2222 sust@38.127.121.195
```

回测命令：

```bash
nc -vz 38.127.121.195 2222
ssh -p 2222 sust@38.127.121.195
```

关键证据摘要：

1. 本机隧道服务：`lanemaster3d-vps-ssh-tunnel.service`，状态为`enabled + active`
2. VPS监听：`0.0.0.0:2222`
3. VPS sshd配置：`gatewayports clientspecified`、`allowtcpforwarding yes`
4. VPS防火墙：`ufw allow 2222/tcp`
5. 实测可登录到`sust-mf`

服务与配置文件位置：

1. 本机服务：`/etc/systemd/system/lanemaster3d-vps-ssh-tunnel.service`
2. VPS配置：`/etc/ssh/sshd_config.d/99-lanemaster3d-tunnel.conf`
3. 固定入口记录：`runtime/ssh_tunnel_vps/connect_command.txt`
4. 辅助脚本：`scripts/remote/show_fixed_ssh_entry.sh`

### 3.2 旧链路历史（保留）

测试命令：

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 -vv sust@111.114.18.212 exit
ssh -o BatchMode=yes -o ConnectTimeout=8 -vv sust@123.139.38.70 exit
Test-NetConnection 111.114.18.212 -Port 22
```

返回结果：

```text
ssh: connect to host 111.114.18.212 port 22: Connection timed out
ssh: connect to host 123.139.38.70 port 22: Connection timed out
TcpTestSucceeded : False
```

结论：服务器`sshd`服务本身正常，但外部网络到服务器公网入口未打通。

补充观测（2026-02-17）：

1. `ssh.service`状态为`active (running)`，已连续运行5天。
2. `sshd`正在监听`0.0.0.0:22`与`[::]:22`。
3. `ufw`状态为`不活动`（未启用）。
4. `/etc/ssh/sshd_config`未显式配置`Port/PasswordAuthentication/AllowUsers`等键（使用默认或include配置）。
5. 已新增`/etc/ssh/sshd_config.d/99-lanemaster3d.conf`，包含`AllowUsers sust`与密码登录配置。
6. 服务器本机已验证：`ssh -vv sust@127.0.0.1`和`ssh -vv sust@111.114.18.212`均可成功认证登录。
7. 外部（Codex执行机）连接在2026-02-18复测返回：`Connection timed out`。
8. 外部执行机`Test-NetConnection 111.114.18.212 -Port 22`结果：`TcpTestSucceeded=False`。
9. 外部执行机清空`HTTP_PROXY/HTTPS_PROXY/ALL_PROXY`后复测，结果仍为`Connection timed out`。
10. 推断为公网链路/NAT映射/边界设备策略导致，非`sshd`进程本身故障。
11. 服务器侧抓包断点：外网`SYN`到达，本机返回`SYN-ACK`，对端未回`ACK`。
12. 服务器侧已建立Pinggy反向隧道：`xfjrs-123-139-38-70.a.free.pinggy.link:39029`。
13. 外部执行机已实测登录成功（密码认证）。

## 3.3 临时可用SSH入口（Pinggy，历史）

连接命令：

```bash
ssh -p 39029 sust@xfjrs-123-139-38-70.a.free.pinggy.link
```

外部执行机验证命令：

```bash
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o ConnectTimeout=10 -p 39029 sust@xfjrs-123-139-38-70.a.free.pinggy.link exit
```

验证结果摘要：

```text
Permission denied (publickey,password,keyboard-interactive)
```

说明：上述`BatchMode=yes`用于无交互探测，会在认证阶段返回拒绝；随后使用交互式密码登录已成功进入服务器Shell。

## 3.4 本机采集命令（在服务器本机终端执行）

```bash
echo "=== identity ==="
whoami
id
hostname -I

echo "=== ssh service ==="
sudo systemctl status ssh --no-pager -l
sudo ss -tlnp | grep ':22'
sudo grep -E '^(Port|PasswordAuthentication|PermitRootLogin|AllowUsers)' /etc/ssh/sshd_config

echo "=== firewall ==="
sudo ufw status verbose

echo "=== gpu ==="
nvidia-smi -L

echo "=== project path ==="
ls -lah /home/sust/zhou/lanemaster3d
ls -lah /home/sust/zhou/lanemaster3d/data
ls -lah /home/sust/zhou/lanemaster3d/data/OpenLane/data_lists
```

## 4. 远端目录核对清单

1. `/home/sust/zhou/lanemaster3d`
2. `/home/sust/zhou/lanemaster3d/data/OpenLane/images`
3. `/home/sust/zhou/lanemaster3d/data/OpenLane/data_lists/training.txt`
4. `/home/sust/zhou/lanemaster3d/data/OpenLane/data_lists/validation.txt`

## 5. 接管后首批执行命令

```bash
cd /home/sust/zhou/lanemaster3d
bash scripts/remote/check_env.sh
bash scripts/remote/check_openlane_data.sh .
bash scripts/remote/start_train_tmux.sh lm3d_train configs/openlane/r50_960x720.py experiments/lm3d_exp001 2
```

## 6. 当前训练状态（2026-02-18 19:24）

1. 代码已同步到远端：`/home/sust/zhou/lanemaster3d`
2. 配置已切换为高吞吐稳定版：
   `amp=True`、`amp_dtype=bfloat16`、`find_unused_parameters=False`、
   `batch_size_per_gpu=20`、`num_workers=10`、`tf32=True`、
   `channels_last=True`、`ddp_static_graph=True`
3. DDP未使用参数问题已修复（`decode_stages.*.cls_layer`纳入`ddp_aux`计算图）。
4. 本地与远端关键测试通过：
   `python -m pytest -q tests/test_lane_master_net.py` => `4 passed`
5. 当前训练会话：`tmux`会话名`lm3d_train`
6. 当前实验目录：`/home/sust/zhou/lanemaster3d/experiments/lm3d_exp020`
7. 指标日志：`/home/sust/zhou/lanemaster3d/experiments/lm3d_exp020/logs/metrics.jsonl`
8. 已观测到训练迭代输出（epoch1 iter20/40/60/80/100），无`non-finite loss`告警。
9. 本轮提速改造（已落地）：
   1. 不确定性损失复用`LaneSetLoss`匹配结果，去掉重复匹配计算。
   2. TopK分配器距离计算改为广播矩阵，去掉`repeat_interleave + cat`大张量展开。
10. 实测吞吐（双卡总样本/秒，按`iter_time`估算）：
   1. `exp015`尾部均值：`4.95 samples/s`（`avg_iter_time≈8.08s`）。
   2. `exp016`前6条均值：`8.09 samples/s`（`avg_iter_time≈4.95s`）。
   3. `exp017`前3条均值：`9.38 samples/s`（`avg_iter_time≈4.26s`）。
   4. `exp020`前6条均值：`10.85 samples/s`（`avg_iter_time≈3.69s`）。
11. 当前GPU利用率（20秒窗口，`nvidia-smi dmon`）：
   1. `GPU0 avg_sm≈23.4%`，`peak≈84%`
   2. `GPU1 avg_sm≈38.0%`，`peak≈100%`
12. 说明：
   1. `torch.compile`在本机首轮编译阶段耗时过长且日志长时间不落盘，当前已回退为`compile_model=False`以保证训练稳定推进。
   2. 当前瓶颈是“脉冲式小算子 + 主机侧调度”，峰值可到90%+，但20秒均值难以稳定在80%。
13. 监控命令：

```bash
cd /home/sust/zhou/lanemaster3d
tmux ls
tmux capture-pane -pt lm3d_train -S -200 | tail -n 120
tail -f /home/sust/zhou/lanemaster3d/experiments/lm3d_exp020/logs/metrics.jsonl
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
ps -ef | grep -E 'tools/train.py|torchrun' | grep -v grep
nvidia-smi dmon -s um -d 1 -c 8
```
