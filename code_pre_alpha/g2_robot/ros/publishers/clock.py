# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""OmniGraph-based clock and real-time factor ROS2 publishers."""

import omni.graph.core as og


def publish_clock(graph_path="/RosClockActionGraph"):
    """Publish simulation clock via ROS2 OmniGraph."""
    og.Controller.edit(
        {
            "graph_path": graph_path,
            "evaluator_name": "execution",
            "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
        },
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("RosContext", "isaacsim.ros2.bridge.ROS2Context"),
                ("RosPublisherClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "RosPublisherClock.inputs:execIn"),
                ("OnPlaybackTick.outputs:time", "RosPublisherClock.inputs:timeStamp"),
                ("RosContext.outputs:context", "RosPublisherClock.inputs:context"),
            ],
        },
    )


def publish_rtf(graph_path="/RosRTFActionGraph"):
    """Publish real-time factor via ROS2 OmniGraph."""
    og.Controller.edit(
        {
            "graph_path": graph_path,
            "evaluator_name": "execution",
            "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
            "fc_backing_type": og.GraphBackingType.GRAPH_BACKING_TYPE_FLATCACHE_SHARED,
            "evaluation_mode": og.GraphEvaluationMode.GRAPH_EVALUATION_MODE_AUTOMATIC,
        },
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("RosContext", "isaacsim.ros2.bridge.ROS2Context"),
                ("RTF", "isaacsim.core.nodes.IsaacRealTimeFactor"),
                ("RosPublisherRTF", "isaacsim.ros2.bridge.ROS2Publisher"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("RosPublisherRTF.inputs:messageName", "Float32"),
                ("RosPublisherRTF.inputs:messagePackage", "std_msgs"),
                ("RosPublisherRTF.inputs:messageSubfolder", "msg"),
                ("RosPublisherRTF.inputs:topicName", "rtf_factor"),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "RosPublisherRTF.inputs:execIn"),
                ("RosContext.outputs:context", "RosPublisherRTF.inputs:context"),
            ],
        },
    )
    og.Controller.connect(
        og.Controller.attribute(graph_path + "/RTF.outputs:rtf"),
        og.Controller.attribute(graph_path + "/RosPublisherRTF.inputs:data"),
    )
