import os
from pathlib import Path
from controller import Supervisor, Node

TIME_STEP = 16


def parse_vec(text, default):
    if not text:
        return default
    return [float(x.strip()) for x in text.split(',')]


def find_viewpoint(supervisor: Supervisor):
    children = supervisor.getRoot().getField('children')
    for i in range(children.getCount()):
        node = children.getMFNode(i)
        if node.getType() == Node.VIEWPOINT:
            return node
    return None


def run_steps(robot: Supervisor, steps: int):
    for _ in range(steps):
        if robot.step(TIME_STEP) == -1:
            return False
    return True


def main():
    supervisor = Supervisor()
    mode = os.environ.get('SWARM_CAPTURE_MODE', 'idle')

    self_node = supervisor.getSelf()
    camera = supervisor.getDevice('overhead_camera')

    if mode == 'idle':
        print('[capture] idle supervisor started')
        while supervisor.step(TIME_STEP) != -1:
            pass
        return

    settle_steps = int(os.environ.get('SWARM_SETTLE_STEPS', '80'))

    if mode.startswith('view_'):
        viewpoint = find_viewpoint(supervisor)
        if viewpoint is None:
            print('[capture] viewpoint not found')
            supervisor.simulationQuit(1)
            return
        position = parse_vec(os.environ.get('SWARM_VIEW_POSITION'), [0.0, 0.0, 2.8])
        orientation = parse_vec(os.environ.get('SWARM_VIEW_ORIENTATION'), [0.0, 0.0, 1.0, 0.0])
        viewpoint.getField('position').setSFVec3f(position)
        viewpoint.getField('orientation').setSFRotation(orientation)
        print(f'[capture] viewpoint position={position} orientation={orientation}')
        if not run_steps(supervisor, settle_steps):
            return
        if mode == 'view_image':
            output = Path(os.environ.get('SWARM_CAPTURE_IMAGE', '/tmp/swarm_capture.png'))
            output.parent.mkdir(parents=True, exist_ok=True)
            supervisor.exportImage(str(output), 100)
            print(f'[capture] exported image to {output}')
            run_steps(supervisor, 10)
            supervisor.simulationQuit(0)
            return
        if mode == 'view_sequence':
            out_dir = Path(os.environ.get('SWARM_CAPTURE_DIR', '/tmp/swarm_view_frames'))
            out_dir.mkdir(parents=True, exist_ok=True)
            frame_count = int(os.environ.get('SWARM_FRAME_COUNT', '120'))
            stride = int(os.environ.get('SWARM_FRAME_STRIDE', '2'))
            for frame_idx in range(frame_count):
                frame_path = out_dir / f'frame_{frame_idx:04d}.png'
                supervisor.exportImage(str(frame_path), 100)
                if not run_steps(supervisor, stride):
                    supervisor.simulationQuit(0)
                    return
            print(f'[capture] exported {frame_count} viewpoint frames to {out_dir}')
            run_steps(supervisor, 10)
            supervisor.simulationQuit(0)
            return

    if mode.startswith('camera_'):
        if camera is None:
            print('[capture] overhead_camera not found')
            supervisor.simulationQuit(1)
            return
        robot_translation = parse_vec(os.environ.get('SWARM_CAMERA_TRANSLATION'), [0.0, 0.0, 2.8])
        robot_rotation = parse_vec(os.environ.get('SWARM_CAMERA_ROTATION'), [0.0, 0.0, 1.0, 0.0])
        self_node.getField('translation').setSFVec3f(robot_translation)
        self_node.getField('rotation').setSFRotation(robot_rotation)
        camera.enable(TIME_STEP)
        print(f'[capture] camera robot translation={robot_translation} rotation={robot_rotation}')
        if not run_steps(supervisor, settle_steps):
            return
        if mode == 'camera_image':
            output = Path(os.environ.get('SWARM_CAPTURE_IMAGE', '/tmp/swarm_camera.png'))
            output.parent.mkdir(parents=True, exist_ok=True)
            camera.saveImage(str(output), 100)
            print(f'[capture] saved camera image to {output}')
            run_steps(supervisor, 10)
            supervisor.simulationQuit(0)
            return
        if mode == 'camera_sequence':
            out_dir = Path(os.environ.get('SWARM_CAPTURE_DIR', '/tmp/swarm_frames'))
            out_dir.mkdir(parents=True, exist_ok=True)
            frame_count = int(os.environ.get('SWARM_FRAME_COUNT', '120'))
            stride = int(os.environ.get('SWARM_FRAME_STRIDE', '2'))
            for frame_idx in range(frame_count):
                frame_path = out_dir / f'frame_{frame_idx:04d}.png'
                camera.saveImage(str(frame_path), 100)
                if not run_steps(supervisor, stride):
                    supervisor.simulationQuit(0)
                    return
            print(f'[capture] saved {frame_count} camera frames to {out_dir}')
            run_steps(supervisor, 10)
            supervisor.simulationQuit(0)
            return

    print(f'[capture] unknown mode: {mode}')
    supervisor.simulationQuit(1)


if __name__ == '__main__':
    main()
