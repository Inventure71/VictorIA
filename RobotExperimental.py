import math
import time

from interbotix_common_modules.common_robot.robot import robot_startup
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS


class RobotHandling:
    def __init__(self):
        self.bot = InterbotixManipulatorXS(
            robot_model='wx250s',
            group_name='arm',
            gripper_name='gripper',
        )

        self.base_speed = 6

        self.current_location = (0, 0, 0)

        self.safe_location = (-0.2, 0.01, 0.3)

        self.vertical_pitch = 1.43

        # coordinates of the tower
        self.tower_coords = (-0.2, 0.01, 0.1)

        # coordinates of the closest cell (0)
        self.cell_0_cord = (0.1, 0.01, 0.23)

        self.distance_between_cells = 0.031

        self.adjustment_multiplier = 0.0025

        robot_startup()

    def move_arm_cartesian(self, x=None, y=None, z=None, roll=None, pitch=None, yaw=None, speed=1.0):
        if x is None:
            x = self.current_location[0]
        elif y is None:
            y = self.current_location[1]
        elif z is None:
            z = self.current_location[2]

        target_location = (x, y, z)

        distance = self.find_distance(self.current_location, target_location)

        if speed is None:
            moving_time = None
        else:
            moving_time = max(2.0, distance / (self.base_speed * speed))

        kwargs = {k: v for k, v in {
            "x": x, "y": y, "z": z,
            "roll": roll, "pitch": pitch, "yaw": yaw,
            "moving_time": moving_time
        }.items() if v is not None}


        self.bot.arm.set_ee_pose_components(**kwargs)
        self.current_location = target_location


    def find_distance(self, start_location, end_location):
        distance = math.sqrt(
            (start_location[0] - end_location[0]) ** 2 +
            (start_location[1] - end_location[1]) ** 2 +
            (start_location[2] - end_location[2]) ** 2
        )
        print("Distance:", distance)
        return distance

    def move_to_safe_location(self):
        # , pitch=self.vertical_pitch
        self.move_arm_cartesian(self.safe_location[0], self.safe_location[1], self.safe_location[2], pitch=self.vertical_pitch/4, speed=1.2)

    def pickup_cell(self):
        print("picking up chip")
        self.move_to_safe_location()

        self.bot.gripper.release(2.0)

        self.move_arm_cartesian(self.tower_coords[0], self.tower_coords[1], self.tower_coords[2] + 0.1,
                                            pitch=self.vertical_pitch)
        time.sleep(0.1)
        self.move_arm_cartesian(self.tower_coords[0], self.tower_coords[1], self.tower_coords[2],
                                            pitch=self.vertical_pitch, speed=0.5)

        self.bot.gripper.grasp(2.0)

        self.move_to_safe_location()
        print("picked up")

    def move_to_cell(self, column: int):
        """
        Moves the arm to drop a piece at the specified column.
        column should be an int from 0 to 6 (for 7 columns).
        """
        print(f"[Robot] Received command to move to column {column}.")

        self.pickup_cell()

        if column != 0:
            # If not that one then move there fast WITH increased height
            self.move_arm_cartesian(self.cell_0_cord[0], self.cell_0_cord[1], self.cell_0_cord[2]+0.05, pitch=self.vertical_pitch)
            time.sleep(1)

        # Compute the target X offset based on the column
        # target_x = 0.18 + int(column) * 0.031

        target_x = self.cell_0_cord[0] + int(column) * self.distance_between_cells
        target_y = self.cell_0_cord[1]
        target_z = self.cell_0_cord[2]

        print(f"[Robot] Moving end effector to X={target_x:.3f}, Z={target_z:.3f}.")
        self.move_arm_cartesian(x=target_x, y=target_y, z=target_z, pitch=self.vertical_pitch, speed=0.3)

        # perform a slight Z adjustment if column < 4
        #if int(column) < 4:
        #    adjustment_z = -(8 - int(column)) * self.adjustment_multiplier
        #    print(f"[Robot] Applying Z offset: {adjustment_z:.4f}")
        #    self.bot.arm.set_ee_cartesian_trajectory(z=adjustment_z)

        time.sleep(0.2)
        # Drop the piece
        self.bot.gripper.release(2.0)
        time.sleep(0.2)

        # Move arm back up
        self.move_to_safe_location()

        print("[Robot] Move completed.\n")

    def stop(self):
        """Move the robot to a 'sleep' or safe pose."""
        print("[Robot] Stopping and going to sleep pose.")
        self.move_to_safe_location()
        self.bot.arm.go_to_sleep_pose()

if __name__ == '__main__':
    robot = RobotHandling()
    while True:
        cell = input("Cell: ").strip()
        if cell == "q":
            break
        else:
            try:
                cell = int(cell)
                robot.move_to_cell(cell)
            except Exception as e:
                print("input was not an int", e)

    robot.stop()
