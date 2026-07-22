<h1 align="center"> Hello-Robotics </h1>

## 🎯 项目介绍

> &emsp;&emsp;*hello-robotics 基于 G2 与 Isaac 构建一站式机器人仿真实践教程平台，覆盖环境搭建、视觉感知、建图定位、导航抓取与大模型决策全流程，支持前沿多模态及具身智能算法落地验证。*

&emsp;&emsp;当前机器人仿真与具身智能领域缺乏从环境搭建、感知建图、导航操作到大模型决策的一体化、可复现、循序渐进的实践教程，现有仿真平台资料零散、门槛高、难以形成完整工程化体系。hello-robotics 基于 Isaac 与 G2 构建全流程仿真实践平台，既能补齐系统化教学的空白，降低机器人开发入门成本，又可为 VLM、VLA 等前沿具身智能算法提供统一验证底座，兼具教学价值与科研实践价值，对推动仿真机器人技术普及与算法落地具有重要意义。

## 🔍 效果展示

<table align="center">
  <tr>
    <td colspan="2" valign="top" align="center">
      <img src="assets/love_you.gif" width="70%">
      <br>
      <strong>底盘机械臂基础控制</strong>
      <br>
      <sub><strong>给大家比个心，希望大家能够喜欢我们的教程</strong></sub>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top" align="center">
      <img src="assets/mapping1.gif" width="100%">
      <br>
      <strong>g2双雷达模块搭建</strong>
      <br>
      <sub>原有g2没有激光雷达发布，因此在g2上加载了两个激光雷达</sub>
    </td>
    <td width="50%" valign="top" align="center">
      <img src="assets/yolo26.gif" width="100%">
      <br>
      <strong>视觉目标检测</strong>
      <br>
      <sub>g2采用yolo26对环境中的物体进行目标检测</sub>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top" align="center">
      <img src="assets/mapping2.gif" width="100%">
      <br>
      <strong>纯手搓双雷达3d建图算法</strong>
      <br>
      <sub>教程编写简单易懂的3d建图算法帮助大家入门3d建图</sub>
    </td>
    <td width="50%" valign="top" align="center">
      <img src="assets/2d_mapping.gif" width="100%">
      <br>
      <strong>纯手搓双雷达2d建图算法</strong>
      <br>
      <sub>教程采用3d建图的双3d激光雷达点云编写2d建图算法</sub>
    </td>
  </tr>

</table>

## 📖 内容导航

| 章节                                                                                        | 关键内容                                      | 状态 |
| ------------------------------------------------------------------------------------------- | --------------------------------------------- | ---- |
| <strong>第一部分 构建虚拟世界：IsaacSim快速入门</strong>                                       |                                               |      |
| [第一章 Isaacsim环境配置与安装](./docs/chapter1/第一章%20Isaacsim环境配置与安装.md)                        |                   | ✅    |
| [第二章 Isaacsim基本使用](./docs/chapter2/第二章%20Isaacsim基本使用.md)                            |              | ✅    |
| [第三章 Isaacsim综合实践](./docs/chapter3/第三章%20Isaacsim综合实践.md)                         |           | 🚧    |
| <strong>第二部分 掌控机械躯体：机器人运动控制实战</strong>                                     |                                               |      |
| 第四章 移动底盘运动学与控制                   | 底盘拆解与线速度角速度控制  | 🚧    |
| [第五章 机械臂运动学与关节控制](./docs/chapter5/第五章%20机械臂运动学与关节控制.md)                | 机械臂关节控制、正逆运动学（FK/IK）求解   | 🚧    |
| 第六章 运动控制综合实践：移动与抓取                    | 结合底盘与机械臂，完成简单轨迹规划与定点抓取演练 | 🚧    |
| <strong>第三部分 接入多维感官：视觉感知与环境理解</strong>                                      |                                               |      |
| 第七章 2D视觉感知                                | 部署 YOLO 等经典算法，实现仿真环境下的物体识别与语义分割  | 🚧    |
| 第八章 3D空间理解                                 | 点云与位姿估计                         | 🚧    |
| 第九章 感知控制综合实践：视觉感知控制                      | 结合视觉感知与机械臂控制，完成基于视觉引导的动态抓取任务              | 🚧    |
| <strong>第四部分 穿梭复杂场景：SLAM建图与自主导航</strong>                                    |                                               |      |
| 第十章 2D/3D SLAM建图                    | 建图算法，扫图并保存环境地图              | 🚧    |
| 第十一章 Nav2导航框架与规划                     | Nav2全局路径规划与局部避障算法部署       | 🚧    |
| 第十二章 导航综合实践：全场景自主巡航                  | 结合建图与导航算法，实现复杂动态环境下的多点巡航与避障              | 🚧    |
| <strong>第五部分 注入智能灵魂：任务规划与具身决策</strong>                                   |                                               |      |
| 第十三章 LLM接入与Prompt工程                     | 调用大模型 API，设计适用于机器人任务的系统提示词与交互逻辑                  | 🚧    |
| 第十四章 场景描述与常识推理                     | 通过视觉感知与地图分析，通过VLM模型进行空间理解                 | 🚧    |
| 第十五章  视觉语言动作模型部署                    | 通过VLA模型实现从视觉输入到机械臂动作的直接输出                 | 🚧    |
| 第十六章 决策闭环综合实践：听指令做任务                      | 打通感知、大模型决策与底层控制，实现感知理解并执行的完整闭环                 | 🚧    |
| <strong>第六部分 走向前沿落地：前沿具身算法部署实践</strong>                                   |                                               |      |
| 第十七章 前沿算法部署1                    |                  | 🚧    |
| 第十八章 前沿算法部署2                    |                  | 🚧    |

## 贡献者名单

| 姓名 | 职责 | 简介 |
| :----| :---- | :---- |
| 李昀迪 | 项目负责人 | 北京科技大学 |
| 陈可为 | 联合项目负责人 | 中国科学院大学 |
| 张天一 | 联合项目负责人 | 北京工业大学 |

## 参与贡献

- 如果你发现了一些问题，可以提Issue进行反馈，如果提完没有人回复你可以联系[保姆团队](https://github.com/datawhalechina/DOPMC/blob/main/OP.md)的同学进行反馈跟进~
- 如果你想参与贡献本项目，可以提Pull Request，如果提完没有人回复你可以联系[保姆团队](https://github.com/datawhalechina/DOPMC/blob/main/OP.md)的同学进行反馈跟进~
- 如果你对 Datawhale 很感兴趣并想要发起一个新的项目，请按照[Datawhale开源项目指南](https://github.com/datawhalechina/DOPMC/blob/main/GUIDE.md)进行操作即可~

## 关注我们

<div align=center>
<p>扫描下方二维码关注公众号：Datawhale</p>
<img src="https://raw.githubusercontent.com/datawhalechina/pumpkin-book/master/res/qrcode.jpeg" width = "180" height = "180">
</div>

## LICENSE

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a><br />本作品采用<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议</a>进行许可。
*注：默认使用CC 4.0协议，也可根据自身项目情况选用其他协议*
