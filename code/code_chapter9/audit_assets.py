"""用 Isaac Sim Python 检查 USD 的传递依赖是否全部在本章内。"""
import json
from config import ROOT, ROBOT_USD


def main():
    from isaacsim import SimulationApp
    # 独立审计入口同样在启动前禁用扩展源码监视。
    app=SimulationApp({'headless':True,'fast_shutdown':True,
                       'extra_args':['--/app/extensions/fsWatcherEnabled=false']})
    try:
        from pxr import UsdUtils
        layers,assets,unresolved=UsdUtils.ComputeAllDependencies(str(ROBOT_USD))
        from pathlib import Path
        files=[x.realPath for x in layers]+list(assets)
        # OmniPBR 是 Isaac 内置材质，和仿真运行时一起提供，不是其他章节资产。
        runtime=[str(x) for x in files if Path(str(x)).name=='OmniPBR.mdl' and '/kit/mdl/core/' in str(x)]
        external=[str(x) for x in files if not Path(str(x)).resolve().is_relative_to(ROOT) and str(x) not in runtime]
        report=dict(files=files,runtime_materials=runtime,unresolved=list(unresolved),external=external,
                    portable=not unresolved and not external)
        (ROOT/'outputs/asset_audit.json').write_text(json.dumps(report,indent=2))
        print('[资产检查]',report,flush=True)
        if not report['portable']: raise RuntimeError('资产存在外部或未解析依赖')
    finally:
        import sys, omni.kit.app
        omni.kit.app.get_app().post_quit(int(sys.exc_info()[0] is not None))
        app.close()

if __name__=='__main__': main()
