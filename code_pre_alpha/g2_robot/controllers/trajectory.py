# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Ruckig-based smooth trajectory generation for joint-space motion."""

from copy import copy

from g2_robot.utils.logger import Logger

logger = Logger()

from ruckig import InputParameter, OutputParameter, Result, Ruckig


class RuckigTrajectory:
    """Generate smooth joint-space trajectories using the Ruckig library.

    Produces time-optimal trajectories respecting velocity, acceleration,
    and jerk limits for each DOF.
    """

    def __init__(self, dof_num, delta_time):
        """
        Args:
            dof_num: number of degrees of freedom
            delta_time: control cycle time in seconds
        """
        self.otg = Ruckig(dof_num, delta_time)
        self.inp = InputParameter(dof_num)
        self.out = OutputParameter(dof_num)
        self.dof_num = dof_num

    def calculate_trajectory(self, current_position, target_position,
                             max_velocity=None, max_acceleration=None, max_jerk=None):
        """Calculate a smooth trajectory from current to target position.

        Args:
            current_position: list of current joint positions
            target_position: list of target joint positions
            max_velocity: per-DOF velocity limits (default: [2000]*dof_num)
            max_acceleration: per-DOF acceleration limits (default: [10]*dof_num)
            max_jerk: per-DOF jerk limits (default: [50]*dof_num)

        Returns:
            list of waypoint positions (each a list of joint values)
        """
        self.inp.current_position = current_position
        self.inp.target_position = target_position
        self.inp.max_velocity = max_velocity or [2000] * self.dof_num
        self.inp.max_acceleration = max_acceleration or [10] * self.dof_num
        self.inp.max_jerk = max_jerk or [50] * self.dof_num

        first_output, out_list = None, []
        res = Result.Working
        while res == Result.Working:
            res = self.otg.update(self.inp, self.out)
            out_list.append(copy(self.out.new_position))
            self.out.pass_to_input(self.inp)
            if not first_output:
                first_output = copy(self.out)
        return out_list

    # Keep backward-compatible alias
    caculate_trajectory = calculate_trajectory
