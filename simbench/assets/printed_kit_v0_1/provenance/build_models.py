#!/usr/bin/env python3
"""TwinGraph prototype CAD (mm), not a mechanically qualified product.
Source: Mayyoungyoung/twingraph main e2cfe0934ead9bf9431f82e5b6a986b26589f472.
Run: python build_models.py --out output_dir
STL uses print orientation. Individual STEP files use upright design orientation.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import cadquery as cq
import numpy as np
import trimesh
COMMIT = 'e2cfe0934ead9bf9431f82e5b6a986b26589f472'

@dataclass(frozen=True)
class Parameters:
    pin_diameter_mm: float = 6.6
    square_hole_side_mm: float = 8.0
    handle_post_diameter_mm: float = 11.0
    handle_bore_diameter_mm: float = 12.2
    holder_bore_diameter_mm: float = 8.8
    def validate(self) -> None:
        if not all(np.isfinite(x) and x > 0 for x in asdict(self).values()):
            raise ValueError('Dimensions must be finite and positive.')
        if not self.pin_diameter_mm < self.square_hole_side_mm < 15:
            raise ValueError('Square hole must clear the pin and retain adequate walls.')
        if not self.handle_post_diameter_mm < self.handle_bore_diameter_mm < 20:
            raise ValueError('Handle bore must clear the post and retain adequate walls.')
        if not self.pin_diameter_mm < self.holder_bore_diameter_mm < 16:
            raise ValueError('Holder must clear pin and retain walls.')

def box(x:float,y:float,z:float,pos=(0,0,0)) -> cq.Workplane:
    return cq.Workplane('XY').box(x,y,z,centered=(True,True,False)).translate(pos)
def cyl(d:float,h:float,pos=(0,0,0)) -> cq.Workplane:
    return cq.Workplane('XY').circle(d/2).extrude(h).translate(pos)
def annulus(od:float,id_:float,h:float) -> cq.Workplane:
    return cq.Workplane('XY').circle(od/2).circle(id_/2).extrude(h)

def rail_segment(length:float) -> cq.Workplane:
    a=box(length,98,12)
    for s in (-1,1):
        a=a.union(box(length,14,24,(0,s*34,6)))
        a=a.union(box(length,12,6,(0,s*27,26)))
    return a.clean()
def guide_base(p:Parameters) -> cq.Workplane:
    a=box(236,98,12)
    for y in (-32,32):
        a=a.cut(box(p.square_hole_side_mm,p.square_hole_side_mm,14,(-92,y,-1)))
    for s in (-1,1):
        a=a.union(box(178,14,24,(20,s*34,6)))
        a=a.union(box(178,12,6,(20,s*27,26)))
    a=a.union(box(18,52,28,(108,0,6)))
    for x in (-107.23,-76.77):
        for y in (-46.23,46.23):
            a=a.union(cyl(8,18,(x,y,5)))
    return a.clean()
def carriage(p:Parameters) -> cq.Workplane:
    a=box(54,46,12)
    # Added bridge fills source's 4 mm gap, retaining boss and contact datums.
    a=a.union(box(44,28,4,(0,0,12)))
    a=a.union(box(44,28,30,(0,0,16)))
    return a.union(cyl(p.handle_post_diameter_mm,24,(0,0,44))).clean()
def end_stop(p:Parameters) -> cq.Workplane:
    a=box(24,86,36)
    for y in (-32,32):
        a=a.cut(box(p.square_hole_side_mm,p.square_hole_side_mm,38,(0,y,-1)))
    return a.union(box(22,26,16,(0,0,33))).clean()
def pin(p:Parameters) -> cq.Workplane:
    # Source local z: tip [-48,-46], shaft [-47,6], head [-1,13].
    a=cyl(5.4,2).union(cyl(p.pin_diameter_mm,53,(0,0,1)))
    return a.union(cyl(18,14,(0,0,47))).clean()
def wipe_body() -> cq.Workplane:
    # Bond 36x24x8 foam UNDER this rigid part: total nominal tool height=73.
    # Added 3 mm backing; tapered stem replaces the source cosmetic connector.
    a=box(36,24,3).union(box(12,14,12))
    taper=(cq.Workplane('XY').workplane(offset=12).rect(12,14)
           .workplane(offset=9).rect(24,28).loft(combine=True))
    return a.union(taper).union(box(24,28,44,(0,0,21))).clean()
def receiver(p:Parameters) -> cq.Workplane:
    return box(70,70,54).cut(box(p.square_hole_side_mm,p.square_hole_side_mm,56,(0,0,-1))).clean()
def marker_notch(a:cq.Workplane,w:float,h:float,t:float) -> cq.Workplane:
    n=cq.Workplane('XY').polyline([(-w/2,h/2),(-w/2+4,h/2),(-w/2,h/2-4)]).close().extrude(t+2).translate((0,0,-1))
    return a.cut(n).clean()
def pin_coupon() -> cq.Workplane:
    a=box(100,44,10)
    for x,d,w in zip((-36,-18,0,18,36),(6.8,7.0,7.2,7.6,8.0),(7.0,7.2,7.6,8.0,8.4)):
        a=a.cut(cyl(d,12,(x,10,-1))).cut(box(w,w,12,(x,-10,-1)))
    return marker_notch(a,100,44,10)
def handle_coupon() -> cq.Workplane:
    a=box(80,24,8)
    for x,d in zip((-27,-9,9,27),(11.4,11.8,12.2,12.6)):
        a=a.cut(cyl(d,10,(x,0,-1)))
    return marker_notch(a,80,24,8)
def make_parts(p:Parameters):
    return {
      '01_guide_base_nominal':(guide_base(p),1,'导轨底座（未设计安装孔）'),
      '02_carriage_bridged':(carriage(p),1,'滑块（已补结构连接）'),
      '03_end_stop':(end_stop(p),1,'端挡（双8 mm方孔）'),
      '04_pin':(pin(p),2,'插销（STL为头朝下）'),
      '05_handle_ring':(annulus(42,p.handle_bore_diameter_mm,16),1,'手柄环'),
      '06_pin_supply_holder':(annulus(20,p.holder_bore_diameter_mm,38),2,'插销供料筒（须固定）'),
      '07_wipe_tool_rigid':(wipe_body(),1,'擦拭工具硬质部分（另加软垫）'),
      '08_single_pin_receiver':(receiver(p),1,'独立插销试验孔座（可选）'),
      '09_cube_40':(box(40,40,40),1,'基础抓取方块（可选）'),
      '10_pin_fit_coupon':(pin_coupon(),1,'插销配合试片'),
      '11_handle_fit_coupon':(handle_coupon(),1,'手柄孔配合试片'),
      '12_post_gauge':(box(26,24,4).union(cyl(p.handle_post_diameter_mm,20,(0,0,4))).clean(),1,'手柄柱测试件'),
      '13_rail_fit_segment':(rail_segment(60),1,'60 mm短导轨试片')}

def export_preview(model, path:Path, width:int=600, height:int=390, direction=(1,-1,0.85)) -> None:
    cq.exporters.export(model,str(path),opt={'width':width,'height':height,
      'marginLeft':width/8,'marginTop':height/8,'projectionDir':direction,
      'showAxes':False,'showHidden':False,'strokeWidth':0.6})
    # OCCT's default projection basis has downward world-Z for these views.
    # Roll the drawing 180 degrees (not a mirror) to present the part upright.
    tree=ET.parse(path); root=tree.getroot()
    group=ET.Element('{http://www.w3.org/2000/svg}g',{'transform':f'rotate(180 {width/2} {height/2})'})
    for child in list(root):
        root.remove(child); group.append(child)
    root.append(group); tree.write(path,encoding='unicode')


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--params',type=Path,help='Optional JSON Parameters overrides')
    args=ap.parse_args()
    p=Parameters(**json.loads(args.params.read_text('utf-8'))) if args.params else Parameters()
    p.validate()
    for d in ('stl','step','preview','validation','source'):
        (args.out/d).mkdir(parents=True,exist_ok=True)
    (args.out/'source/parameters.json').write_text(json.dumps(asdict(p),indent=2)+'\n')
    parts=make_parts(p); results=[]
    for key,(model,qty,label) in parts.items():
        s=model.val()
        if not s.isValid() or len(model.solids().vals())!=1:
            raise ValueError(f'Expected exactly one valid solid: {key}')
        spath=args.out/'step'/f'{key}.step'
        cq.exporters.export(model,str(spath))
        oriented=model.rotate((0,0,0),(1,0,0),180) if key.startswith('04_') else model
        oriented=oriented.translate((0,0,-oriented.val().BoundingBox().zmin))
        stlpath=args.out/'stl'/f'{key}.stl'
        cq.exporters.export(oriented,str(stlpath),tolerance=0.015,angularTolerance=0.06)
        mesh=trimesh.load_mesh(stlpath,process=True); rt=cq.importers.importStep(str(spath))
        ok=bool(mesh.is_watertight and mesh.is_winding_consistent and mesh.is_volume and mesh.body_count==1
                and abs(mesh.bounds[0,2])<1e-5 and rt.val().isValid() and len(rt.solids().vals())==1)
        record=dict(id=key,name_zh=label,quantity=qty,units='mm',print_bbox_mm=mesh.extents.round(4).tolist(),
          cad_valid=s.isValid(),cad_solids=1,stl_watertight=bool(mesh.is_watertight),
          stl_winding_consistent=bool(mesh.is_winding_consistent),stl_body_count=int(mesh.body_count),
          step_roundtrip_valid=rt.val().isValid(),positive_mesh_volume=bool(mesh.volume>0),
          triangles=len(mesh.faces),solid_volume_mm3=round(s.Volume(),3),pass_geometry=ok)
        results.append(record)
        print(key,record['print_bbox_mm'],'PASS' if ok else 'FAIL',flush=True)
        if not ok: raise RuntimeError(f'Export validation failed: {key}')
        export_preview(model,args.out/'preview'/f'{key}.svg')
    asm=cq.Assembly(name='TwinGraph_nominal_prototype_NOT_PRINT_LAYOUT')
    placements={
      'base':('01_guide_base_nominal',(0,0,0)),
      'carriage':('02_carriage_bridged',(30,0,12)),
      'stop':('03_end_stop',(-92,0,12)),
      'pin_left':('04_pin',(-92,-32,1)),
      'pin_right':('04_pin',(-92,32,1)),
      'handle':('05_handle_ring',(30,0,58))}
    world_shapes={}
    for name,(key,pos) in placements.items():
        w=parts[key][0].translate(pos); asm.add(w,name=name); world_shapes[name]=w
    asm.export(str(args.out/'step/ASSEMBLY_VIEW_ONLY.step'))
    comp=cq.Compound.makeCompound([w.val() for w in world_shapes.values()])
    export_preview(comp,args.out/'preview/assembly.svg',1100,620,(-1,-1,0.9))
    overlaps=[]; ns=list(world_shapes)
    for i,a in enumerate(ns):
        for b in ns[i+1:]:
            vol=world_shapes[a].val().intersect(world_shapes[b].val()).Volume()
            if vol>1e-6: overlaps.append(dict(a=a,b=b,intersection_volume_mm3=vol))
    stroke=[]
    for x in np.linspace(-40,60,21):
        w=parts['02_carriage_bridged'][0].translate((float(x),0,12))
        for fixed in ('base','stop','pin_left','pin_right'):
            v=w.val().intersect(world_shapes[fixed].val()).Volume()
            if v>1e-6: stroke.append(dict(x_mm=float(x),fixed=fixed,intersection_volume_mm3=v))
    summary=dict(source_commit=COMMIT,cadquery_version=cq.__version__,trimesh_version=trimesh.__version__,
      scope='Nominal geometry only; no slicing, printing, strength, friction, vision or robot validation.',
      parts=results,assembly_nominal_overlaps=overlaps,
      carriage_stroke_static_checks=dict(x_range_mm=[-40,60],samples=21,overlaps=stroke),
      all_export_checks_passed=all(r['pass_geometry'] for r in results))
    (args.out/'validation/geometry_checks.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n','utf-8')
    print('Assembly intersections:',overlaps,flush=True)
    print('Stroke intersections:',stroke,flush=True)
    if overlaps or stroke: raise RuntimeError('Nominal assembly interference requires investigation')
if __name__=='__main__': main()
