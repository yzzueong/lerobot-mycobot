########################################################################################
# Utilities
########################################################################################


import logging
import threading
import time
import traceback
from contextlib import nullcontext
from copy import copy
from functools import cache

import cv2
import keyboard as kb
import numpy as np
import torch
from deepdiff import DeepDiff
from termcolor import colored

from lerobot.common.datasets.image_writer import safe_stop_image_writer
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import get_features_from_robot
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.utils.utils import get_safe_torch_device, has_method


def log_control_info(robot: Robot, dt_s, episode_index=None, frame_index=None, fps=None):
    log_items = []
    if episode_index is not None:
        log_items.append(f"ep:{episode_index}")
    if frame_index is not None:
        log_items.append(f"frame:{frame_index}")

    def log_dt(shortname, dt_val_s):
        nonlocal log_items, fps
        info_str = f"{shortname}:{dt_val_s * 1000:5.2f} ({1 / dt_val_s:3.1f}hz)"
        if fps is not None:
            actual_fps = 1 / dt_val_s
            if actual_fps < fps - 1:
                info_str = colored(info_str, "yellow")
        log_items.append(info_str)

    # total step time displayed in milliseconds and its frequency
    log_dt("dt", dt_s)

    # TODO(aliberts): move robot-specific logs logic in robot.print_logs()
    if not robot.robot_type.startswith("stretch"):
        for name in robot.leader_arms:
            key = f"read_leader_{name}_pos_dt_s"
            if key in robot.logs:
                log_dt("dtRlead", robot.logs[key])

        for name in robot.follower_arms:
            key = f"write_follower_{name}_goal_pos_dt_s"
            if key in robot.logs:
                log_dt("dtWfoll", robot.logs[key])

            key = f"read_follower_{name}_pos_dt_s"
            if key in robot.logs:
                log_dt("dtRfoll", robot.logs[key])

        for name in robot.cameras:
            key = f"read_camera_{name}_dt_s"
            if key in robot.logs:
                log_dt(f"dtR{name}", robot.logs[key])

    info_str = " ".join(log_items)
    logging.info(info_str)


@cache
def is_headless():
    """Detects if python is running without a monitor."""
    try:
        import pynput  # noqa

        return False
    except Exception:
        print(
            "Error trying to import pynput. Switching to headless mode. "
            "As a result, the video stream from the cameras won't be shown, "
            "and you won't be able to change the control flow with keyboards. "
            "For more info, see traceback below.\n"
        )
        traceback.print_exc()
        print()
        return True


def predict_action(observation, policy, device, use_amp):
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        for name in observation:
            if "image" in name:
                observation[name] = observation[name].type(torch.float32) / 255
                observation[name] = observation[name].permute(2, 0, 1).contiguous()
            observation[name] = observation[name].unsqueeze(0)
            observation[name] = observation[name].to(device)

        # Compute the next action with the policy
        # based on the current observation
        action = policy.select_action(observation)

        # Remove batch dimension
        action = action.squeeze(0)

        # Move to cpu, if not already the case
        action = action.to("cpu")

    return action


def init_keyboard_listener():
    # Allow to exit early while recording an episode or resetting the environment,
    # by tapping the right arrow key '->'. This might require a sudo permission
    # to allow your terminal to monitor keyboard events.
    events = {}
    events["exit_early"] = False
    events["rerecord_episode"] = False
    events["stop_recording"] = False

    if is_headless():
        logging.warning(
            "Headless environment detected. On-screen cameras display and keyboard inputs will not be available."
        )
        listener = None
        return listener, events

    # Only import pynput if not in a headless environment
    from pynput import keyboard

    def on_press(key):
        try:
            if key == keyboard.Key.right:
                print("Right arrow key pressed. Exiting loop...")
                events["exit_early"] = True
            elif key == keyboard.Key.left:
                print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
                events["rerecord_episode"] = True
                events["exit_early"] = True
            elif key == keyboard.Key.esc:
                print("Escape key pressed. Stopping data recording...")
                events["stop_recording"] = True
                events["exit_early"] = True
        except Exception as e:
            print(f"Error handling key press: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    return listener, events


def warmup_record(
    robot,
    events,
    enable_teleoperation,
    warmup_time_s,
    display_cameras,
    fps,
):
    # control_loop(
    #     robot=robot,
    #     control_time_s=warmup_time_s,
    #     display_cameras=display_cameras,
    #     events=events,
    #     fps=fps,
    #     teleoperate=enable_teleoperation,
    # )
    control_loop_mycobot(
        robot=robot,
        control_time_s=warmup_time_s,
        display_cameras=display_cameras,
        events=events,
        fps=fps,
        teleoperate=enable_teleoperation,
    )


def record_episode(
    robot,
    dataset,
    events,
    episode_time_s,
    display_cameras,
    policy,
    device,
    use_amp,
    fps,
    single_task,
):
    # control_loop(
    #     robot=robot,
    #     control_time_s=episode_time_s,
    #     display_cameras=display_cameras,
    #     dataset=dataset,
    #     events=events,
    #     policy=policy,
    #     device=device,
    #     use_amp=use_amp,
    #     fps=fps,
    #     teleoperate=policy is None,
    #     single_task=single_task,
    # )
    control_loop_mycobot(
        robot=robot,
        control_time_s=episode_time_s,
        display_cameras=display_cameras,
        dataset=dataset,
        events=events,
        policy=policy,
        device=device,
        use_amp=use_amp,
        fps=fps,
        teleoperate=policy is None,
        single_task=single_task,
        record=True,
    )


@safe_stop_image_writer
def control_loop(
    robot,
    control_time_s=None,
    teleoperate=False,
    display_cameras=False,
    dataset: LeRobotDataset | None = None,
    events=None,
    policy=None,
    device: torch.device | str | None = None,
    use_amp: bool | None = None,
    fps: int | None = None,
    single_task: str | None = None,
):
    # TODO(rcadene): Add option to record logs
    if not robot.is_connected:
        robot.connect()

    if events is None:
        events = {"exit_early": False}

    if control_time_s is None:
        control_time_s = float("inf")

    if teleoperate and policy is not None:
        raise ValueError("When `teleoperate` is True, `policy` should be None.")

    if dataset is not None and single_task is None:
        raise ValueError("You need to provide a task as argument in `single_task`.")

    if dataset is not None and fps is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset['fps']} != {fps}).")

    if isinstance(device, str):
        device = get_safe_torch_device(device)

    timestamp = 0
    start_episode_t = time.perf_counter()
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if teleoperate:
            observation, action = robot.teleop_step(record_data=True)
        else:
            observation = robot.capture_observation()

            if policy is not None:
                pred_action = predict_action(observation, policy, device, use_amp)
                # Action can eventually be clipped using `max_relative_target`,
                # so action actually sent is saved in the dataset.
                action = robot.send_action(pred_action)
                action = {"action": action}

        if dataset is not None:
            frame = {**observation, **action, "task": single_task}
            dataset.add_frame(frame)

        if display_cameras and not is_headless():
            image_keys = [key for key in observation if "image" in key]
            for key in image_keys:
                cv2.imshow(key, cv2.cvtColor(observation[key].numpy(), cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        if fps is not None:
            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / fps - dt_s)

        dt_s = time.perf_counter() - start_loop_t
        # log_control_info(robot, dt_s, fps=fps)

        timestamp = time.perf_counter() - start_episode_t
        if events["exit_early"]:
            events["exit_early"] = False
            break


@safe_stop_image_writer
def control_loop_mycobot(
    robot,
    control_time_s=None,
    teleoperate=False,
    display_cameras=False,
    dataset: LeRobotDataset | None = None,
    events=None,
    policy=None,
    device: torch.device | str | None = None,
    use_amp: bool | None = None,
    fps: int | None = None,
    single_task: str | None = None,
    record=False,
):
    # TODO(rcadene): Add option to record logs
    if not robot.is_connected:
        robot.connect()

    if events is None:
        events = {"exit_early": False}

    if control_time_s is None:
        control_time_s = float("inf")

    if teleoperate and policy is not None:
        raise ValueError("When `teleoperate` is True, `policy` should be None.")

    if dataset is not None and single_task is None:
        raise ValueError("You need to provide a task as argument in `single_task`.")

    if dataset is not None and fps is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset['fps']} != {fps}).")

    if isinstance(device, str):
        device = get_safe_torch_device(device)

    timestamp = 0
    start_episode_t = time.perf_counter()

    if teleoperate:
        if record:
            # collect_data

            # first step: drag robot arm to do the task
            record_list = []
            recording = True
            robot.mc.release_all_servos(1)

            def _record():
                while recording:
                    angles, gripper = None, None
                    while not angles or not gripper:
                        angles = robot.mc.get_angles()
                        gripper = robot.mc.get_gripper_value()
                    gripper = 1 if gripper <= robot.gripper_open_close_threshold else 0

                    if angles:
                        record_list.append(angles + [gripper])
                        time.sleep(0.1)

            print("start recording action. Press s to stop recording...")
            start_t = time.time()
            record_t = threading.Thread(target=_record, daemon=True)
            record_t.start()
            kb.wait('s')
            recording = False
            record_t.join()
            end_t = time.time()
            print(f"stop recording action. time duration {end_t - start_t}s. record points count {len(record_list)}")

            # second step: replay the trajectory and collect data
            print("start replay action and collect data...")
            for pre_angles, after_angles in zip(record_list, record_list[1:]):
                start_loop_t = time.perf_counter()
                robot.mc.send_angles(pre_angles[:-1], 40)  # joints
                robot.mc.set_gripper_state(pre_angles[-1], 40)  # gripper

                state = torch.from_numpy(np.array(pre_angles)).to(torch.float32)
                after_action = torch.from_numpy(np.array(after_angles)).to(torch.float32)

                images = {}
                for name in robot.cameras:
                    before_camread_t = time.perf_counter()
                    images[name] = robot.cameras[name].async_read()
                    images[name] = torch.from_numpy(images[name])
                    robot.logs[f"read_camera_{name}_dt_s"] = robot.cameras[name].logs["delta_timestamp_s"]
                    robot.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

                # Populate output dictionaries
                observation, action = {}, {}
                observation["observation.state"] = state
                action["action"] = after_action
                for name in robot.cameras:
                    observation[f"observation.images.{name}"] = images[name]
                if dataset is not None:
                    frame = {**observation, **action, "task": single_task}
                    dataset.add_frame(frame)
                if display_cameras and not is_headless():
                    image_keys = [key for key in observation if "image" in key]
                    for key in image_keys:
                        cv2.imshow(key, cv2.cvtColor(observation[key].numpy(), cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)
                if fps is not None:
                    dt_s = time.perf_counter() - start_loop_t
                    busy_wait(1 / fps - dt_s)
                # timestamp = time.perf_counter() - start_episode_t
                if events["exit_early"]:
                    events["exit_early"] = False
                    break
            else:
                while timestamp < control_time_s:
                    start_loop_t = time.perf_counter()
                    # set to origianl position
                    helper_list = [0,0,0,0,0,0,0]
                    robot.mc.send_angles(helper_list[:-1], 40)
                    robot.mc.set_gripper_state(helper_list[-1], 40)
                    state = torch.from_numpy(np.array(helper_list)).to(torch.float32)
                    after_action = torch.from_numpy(np.array(helper_list)).to(torch.float32)
                    images = {}
                    for name in robot.cameras:
                        before_camread_t = time.perf_counter()
                        images[name] = robot.cameras[name].async_read()
                        images[name] = torch.from_numpy(images[name])
                        robot.logs[f"read_camera_{name}_dt_s"] = robot.cameras[name].logs["delta_timestamp_s"]
                        robot.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

                    # Populate output dictionaries
                    observation, action = {}, {}
                    observation["observation.state"] = state
                    action["action"] = after_action
                    for name in robot.cameras:
                        observation[f"observation.images.{name}"] = images[name]
                    if dataset is not None:
                        frame = {**observation, **action, "task": single_task}
                        dataset.add_frame(frame)
                    if display_cameras and not is_headless():
                        image_keys = [key for key in observation if "image" in key]
                        for key in image_keys:
                            cv2.imshow(key, cv2.cvtColor(observation[key].numpy(), cv2.COLOR_RGB2BGR))
                        cv2.waitKey(1)
                    if fps is not None:
                        dt_s = time.perf_counter() - start_loop_t
                        busy_wait(1 / fps - dt_s)
                    timestamp = time.perf_counter() - start_episode_t
                    if events["exit_early"]:
                        events["exit_early"] = False
                        break
    else:
        # inference
        while timestamp < control_time_s:
            start_loop_t = time.perf_counter()
            state, gripper_value = None, None
            while not state or not gripper_value:
                state = robot.mc.get_angles()
                gripper_value = robot.mc.get_gripper_value()
            print("state ", state)
            gripper = 1 if gripper_value <= 50 else 0
            state = torch.from_numpy(np.array(state + [gripper])).to(torch.float32)

            images = {}
            for name in robot.cameras:
                before_camread_t = time.perf_counter()
                images[name] = robot.cameras[name].async_read()
                images[name] = torch.from_numpy(images[name])
                robot.logs[f"read_camera_{name}_dt_s"] = robot.cameras[name].logs["delta_timestamp_s"]
                robot.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

            # Populate output dictionaries and format to pytorch
            observation = {}
            observation["observation.state"] = state
            for name in robot.cameras:
                observation[f"observation.images.{name}"] = images[name]

            if policy is not None:
                pred_action = predict_action(observation, policy, device, use_amp)
                # Action can eventually be clipped using `max_relative_target`,
                # so action actually sent is saved in the dataset.
                # move robot to pred_action
                action = robot.send_action(pred_action)
                action = {"action": action}
            if dataset is not None:
                frame = {**observation, **action, "task": single_task}
                dataset.add_frame(frame)
            if display_cameras and not is_headless():
                image_keys = [key for key in observation if "image" in key]
                for key in image_keys:
                    cv2.imshow(key, cv2.cvtColor(observation[key].numpy(), cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)
            if fps is not None:
                dt_s = time.perf_counter() - start_loop_t
                busy_wait(1 / fps - dt_s)
            timestamp = time.perf_counter() - start_episode_t
            if events["exit_early"]:
                events["exit_early"] = False
                break


def reset_environment(robot, events, reset_time_s, fps):
    # TODO(rcadene): refactor warmup_record and reset_environment
    if has_method(robot, "teleop_safety_stop"):
        robot.teleop_safety_stop()
    #
    # control_loop(
    #     robot=robot,
    #     control_time_s=reset_time_s,
    #     events=events,
    #     fps=fps,
    #     teleoperate=True,
    # )
    robot.mc.send_angles([0,0,0,0,0,0], 40)
    robot.mc.set_gripper_state(0, 40)
    control_loop_mycobot(
        robot=robot,
        control_time_s=reset_time_s,
        events=events,
        fps=fps,
        teleoperate=True,
    )


def stop_recording(robot, listener, display_cameras):
    robot.disconnect()

    if not is_headless():
        if listener is not None:
            listener.stop()

        if display_cameras:
            cv2.destroyAllWindows()


def sanity_check_dataset_name(repo_id, policy_cfg):
    _, dataset_name = repo_id.split("/")
    # either repo_id doesnt start with "eval_" and there is no policy
    # or repo_id starts with "eval_" and there is a policy

    # Check if dataset_name starts with "eval_" but policy is missing
    if dataset_name.startswith("eval_") and policy_cfg is None:
        raise ValueError(
            f"Your dataset name begins with 'eval_' ({dataset_name}), but no policy is provided ({policy_cfg.type})."
        )

    # Check if dataset_name does not start with "eval_" but policy is provided
    if not dataset_name.startswith("eval_") and policy_cfg is not None:
        raise ValueError(
            f"Your dataset name does not begin with 'eval_' ({dataset_name}), but a policy is provided ({policy_cfg.type})."
        )


def sanity_check_dataset_robot_compatibility(
    dataset: LeRobotDataset, robot: Robot, fps: int, use_videos: bool
) -> None:
    fields = [
        ("robot_type", dataset.meta.robot_type, robot.robot_type),
        ("fps", dataset.fps, fps),
        ("features", dataset.features, get_features_from_robot(robot, use_videos)),
    ]

    mismatches = []
    for field, dataset_value, present_value in fields:
        diff = DeepDiff(dataset_value, present_value, exclude_regex_paths=[r".*\['info'\]$"])
        if diff:
            mismatches.append(f"{field}: expected {present_value}, got {dataset_value}")

    if mismatches:
        raise ValueError(
            "Dataset metadata compatibility check failed with mismatches:\n" + "\n".join(mismatches)
        )
