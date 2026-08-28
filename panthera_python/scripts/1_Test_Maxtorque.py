#!/usr/bin/env python3
"""
简单的六关节机器人位置速度控制程序
直接在代码中修改目标位置数组来控制机器人
"""
import select
import sys
import termios
import threading
import time
import tty
from datetime import datetime
from pathlib import Path
from Panthera_lib import Panthera


RECORDINGS_DIR = Path(__file__).resolve().parent / "recordings"


def d435_preview(record_event, stop_event, started_event):
    """显示 D435 彩色流；收到 s 后开始写入 MP4。"""
    try:
        import cv2
        import numpy as np
        import pyrealsense2 as rs
    except ImportError as exc:
        print(f"D435 依赖缺失，无法启动预览：{exc}")
        started_event.set()
        return

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    writer = None
    try:
        pipeline.start(config)
        started_event.set()
        cv2.namedWindow("D435 color preview", cv2.WINDOW_NORMAL)
        print("D435 彩色预览已打开；到达 pos1 后按 s 开始闭合并录制。")
        while not stop_event.is_set():
            frame = pipeline.wait_for_frames().get_color_frame()
            if not frame:
                continue
            image = np.asanyarray(frame.get_data()).copy()
            if record_event.is_set() and writer is None:
                RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
                output = RECORDINGS_DIR / (
                    f"d435_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                )
                writer = cv2.VideoWriter(
                    str(output), cv2.VideoWriter_fourcc(*"mp4v"), 30.0,
                    (image.shape[1], image.shape[0]),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"无法创建 MP4 文件：{output}")
                print(f"D435 录制已开始：{output}")
            if writer is not None:
                writer.write(image)
                label = "REC"
            else:
                label = "预览"
            cv2.putText(image, label, (16, 32), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 0, 255) if writer else (0, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.imshow("D435 color preview", image)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("s"), ord("S")): 
                record_event.set()
            elif key in (ord("q"), ord("Q"), 27):
                stop_event.set()
    except Exception as exc:
        print(f"D435 预览/录制异常：{exc}")
        stop_event.set()
    finally:
        if writer is not None:
            writer.release()
        try:
            pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()


def main():
    record_event = threading.Event()
    stop_event = threading.Event()
    camera_started = threading.Event()
    camera_thread = threading.Thread(
        target=d435_preview,
        args=(record_event, stop_event, camera_started),
        daemon=True,
    )
    camera_thread.start()
    camera_started.wait(timeout=5.0)

    print("\n发送控制命令，移动到 pos1...")
    success = robot.Joint_Pos_Vel(pos1, vel, max_torque, iswait=True)
    print(f"移动状态：{success}")

    # 夹爪先完全打开；之后循环发送 pos1，使机械臂持续保持该位置。
    robot.gripper_open(pos=1.6, vel=0.5, max_tqu=0.5)
    hold_vel = [0.0] * robot.motor_count
    print("已到达 pos1，夹爪已打开。按 s 开始闭合夹爪，按 Ctrl+C 退出。")

    old_settings = None
    if sys.stdin.isatty():
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

    try:
        closed = False
        while not stop_event.is_set():
            # 位置速度模式下重复发送目标，避免保持期间目标超时或通信中断。
            robot.Joint_Pos_Vel(pos1, hold_vel, max_torque, iswait=False)

            if not closed:
                key = None
                if sys.stdin.isatty():
                    readable, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if readable:
                        key = sys.stdin.read(1).lower()
                else:
                    # 非交互终端（例如重定向输入）退回到标准 input 行为。
                    key = input().strip().lower()[:1]

                if key == "s":
                    record_event.set()

                if record_event.is_set():
                    print("收到 s，开始闭合夹爪（最大力矩 0.6 N·m）。")
                    robot.gripper_close(pos=0.0, vel=0.5, max_tqu=0.6)
                    closed = True
                    print("夹爪闭合命令已发送，继续保持 pos1。按 Ctrl+C 退出。")
            else:
                # 持续发送闭合目标，确保夹爪保持位置和最大力矩设置。
                robot.gripper_control(pos=0.0, vel=0.5, max_tqu=0.6)
                time.sleep(0.1)
    finally:
        if old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        stop_event.set()
        camera_thread.join(timeout=2.0)

if __name__ == "__main__":
    robot = Panthera()
    zero_pos = [0.0] * robot.motor_count
    pos1 = [0.0, 0.0, 0.0, 0.0, 0.0, -1.5] 
    vel = [0.5] * robot.motor_count      
    max_torque = [21.0, 36.0, 36.0, 21.0, 10.0, 10.0] 
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被中断")
    except Exception as e:
        print(f"\n错误: {e}")
    finally:
        print("\n\n所有电机已停止")
