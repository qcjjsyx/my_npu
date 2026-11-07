"""Generate the first native draw.io circuit draft from IP_Design.md.

Draw.io is the source of truth; per-page files are export/QA intermediates.
"""
from pathlib import Path
import copy
import shutil
import xml.etree.ElementTree as ET
from html import escape

BASE = Path(__file__).resolve().parents[2]
TARGET = BASE / 'assets/diagrams/source/各模块电路图.drawio'
OUT = BASE / 'assets/diagrams/generated/module_circuits'
OUT.mkdir(parents=True, exist_ok=True)
BACKUP = BASE / 'archive/snapshots/各模块电路图_绘制前备份.drawio'
if not BACKUP.exists():
    shutil.copy2(TARGET, BACKUP)
old = ET.parse(BACKUP).getroot()
doc = ET.Element('mxfile', host='Electron', pages='6', version='29.0.0')

class Page:
    def __init__(self, index, title, subtitle, width=1900, height=1000):
        previous = old.findall('diagram')[index]
        self.diagram = ET.SubElement(doc, 'diagram', name=previous.get('name'), id=previous.get('id'))
        graph = ET.SubElement(self.diagram, 'mxGraphModel', dx=str(width), dy=str(height),
            grid='1', gridSize='10', guides='1', tooltips='1', connect='1', arrows='1',
            fold='1', page='1', pageScale='1', pageWidth=str(width), pageHeight=str(height),
            math='0', shadow='0', background='#FFFFFF')
        self.root = ET.SubElement(graph, 'root')
        ET.SubElement(self.root, 'mxCell', id='0')
        ET.SubElement(self.root, 'mxCell', id='1', parent='0')
        self.n = 0
        self.boxes = {}
        self.text('title', title, 50, 30, width-100, 50, 28, bold=True)
        self.text('subtitle', subtitle, 50, 90, width-100, 35, 16, color='#56657A')
    def node(self, id, label, x, y, w, h, kind='logic', size=18, extra=''):
        styles = {
            'logic': 'rounded=0;fillColor=#FFFFFF;strokeColor=#65788A;',
            'reg': 'rounded=1;arcSize=14;fillColor=#EAF3FF;strokeColor=#2F80ED;',
            'op': 'shape=ellipse;perimeter=ellipsePerimeter;fillColor=#FFFFFF;strokeColor=#172033;',
            'fusion': 'shape=ellipse;perimeter=ellipsePerimeter;fillColor=#FDE8E8;strokeColor=#C65353;',
            'mux': 'shape=trapezoid;direction=east;fillColor=#EEF4F8;strokeColor=#65788A;',
            'port': 'rounded=0;fillColor=#EEF4F8;strokeColor=#D6E0EA;',
            'control': 'rounded=0;fillColor=#F0FAFA;strokeColor=#00A6A6;',
        }
        if kind in ('op','fusion'):
            assert w == h
            size = 32
        c = ET.SubElement(self.root, 'mxCell', id=id, value=escape(label).replace('\n','<br>'), vertex='1', parent='1',
            style=styles[kind]+f'whiteSpace=wrap;html=1;fontFamily=PingFang SC;fontSize={size};fontColor=#172033;strokeWidth=1.6;align=center;verticalAlign=middle;spacing=8;'+extra)
        ET.SubElement(c, 'mxGeometry', x=str(x), y=str(y), width=str(w), height=str(h), **{'as':'geometry'})
        self.boxes[id] = (x,y,w,h,kind)
        return id
    def text(self, id, label, x,y,w,h,size=15,bold=False,color='#56657A'):
        c = ET.SubElement(self.root, 'mxCell', id=id, value=escape(label).replace('\n','<br>'), vertex='1', parent='1',
            style=f'text;html=1;strokeColor=none;fillColor=none;whiteSpace=wrap;fontFamily=PingFang SC;fontSize={size};fontColor={"#172033" if bold else color};fontStyle={1 if bold else 0};align=left;verticalAlign=middle;spacing=0;')
        ET.SubElement(c, 'mxGeometry', x=str(x), y=str(y), width=str(w), height=str(h), **{'as':'geometry'})
    def edge(self, source, target, points=(), exit=(1,.5), entry=(0,.5), label='', control=False):
        assert source in self.boxes and target in self.boxes, (source,target)
        self.n += 1
        c = ET.SubElement(self.root, 'mxCell', id=f'edge-{self.n}', value=label, edge='1', parent='1',
            source=source, target=target,
            style='edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=0;endArrow=block;endFill=1;strokeWidth=1.8;'+
            f'strokeColor={"#00A6A6" if control else "#334E68"};dashed={1 if control else 0};fontFamily=PingFang SC;fontSize=14;fontColor=#56657A;labelBackgroundColor=#FFFFFF;'+
            f'exitX={exit[0]};exitY={exit[1]};exitDx=0;exitDy=0;entryX={entry[0]};entryY={entry[1]};entryDx=0;entryDy=0;')
        g = ET.SubElement(c, 'mxGeometry', relative='1', **{'as':'geometry'})
        if points:
            a = ET.SubElement(g,'Array', **{'as':'points'})
            for x,y in points:
                ET.SubElement(a,'mxPoint',x=str(x),y=str(y))
        return c
    def footer(self, label, y):
        self.text('footer',label,50,y,1750,45,15)

# 1. Entry preparation. No new compute stage is implied by the holding register.
p=Page(0,'入口转换与操作数准备','每个物理前端接收一组 A/B；共 16 组',1900,970)
p.node('blBXtvAJsJPgmYkJTYy3-1','A / B\n原始编码\n16bit',50,200,140,500,'port')
for lane,y,src,dst,bits in [(0,220,'FP16 / BF16','FP19','signed12'),(1,440,'E4M3 / E5M2','FP9','signed5')]:
    p.node(f'extract{lane}',src+'\n字段提取',250,y,200,90)
    p.node(f'class{lane}','源格式分类\n特殊值替换',510,y,200,90)
    p.node(f'expand{lane}',dst+' 字段扩展\nH / Eeff',770,y,200,90)
    p.node(f'complement{lane}','符号 → 补码\n'+bits,1030,y,180,90)
    p.edge('blBXtvAJsJPgmYkJTYy3-1',f'extract{lane}',exit=(1,(y+45-200)/500))
    for a,b in [('extract','class'),('class','expand'),('expand','complement')]:
        p.edge(f'{a}{lane}',f'{b}{lane}')
p.node('int','INT16 / INT8 / INT4\n整数解包与符号扩展',510,670,290,90)
p.edge('blBXtvAJsJPgmYkJTYy3-1','int',points=[(220,715)],exit=(1,.9))
p.node('mode','模式\n选择',1300,365,100,265,'mux')
p.edge('complement0','mode',points=[(1250,265),(1250,410)],entry=(0,.17))
p.edge('complement1','mode',entry=(0,.47))
p.edge('int','mode',points=[(1240,715),(1240,585)],entry=(0,.83))
p.node('hold','输入保持寄存器\nA / B payload\nmetadata',1480,400,210,195,'reg')
p.edge('mode','hold')
p.node('out','→ MUL\n26bit / operand',1760,440,120,110,'port',16)
p.edge('hold','out')
p.node('mode_ctrl','mode',1290,210,120,60,'control')
p.edge('mode_ctrl','mode',exit=(.5,1),entry=(.5,0),control=True)
p.node('meta','mode · bias · scale\nfirst · tag · valid',980,820,300,80,'control',16)
p.edge('meta','hold',points=[(1585,860)],entry=(.5,1),control=True)
p.footer('源格式处理：NaN / ±0 → 0；Inf → 有符号最大有限值。整数通路旁路浮点转换。',915)

# 2. Shared multiplier front end; INT16 products remain separate here.
p=Page(1,'MUL：共享乘法前端','下图为 1 个物理前端；共 16 个相同前端',1700,940)
p.node('in','A / B payload\n26bit',50,335,160,270,'port')
p.node('route','格式解包\nINT16 高低位拆分\n模式 / phase 路由',290,335,240,270)
p.edge('in','route')
for lane,y,bits,pbits in [(0,310,12,24),(1,600,9,18)]:
    p.node(f'operand{lane}',f'A / B\nsigned{bits}',650,y,150,100,'mux')
    p.node(f'mul{lane}','×',920,y+10,80,80,'fusion')
    p.node(f'prod{lane}',f'乘积寄存器\nsigned{pbits}',1100,y-15,190,130,'reg')
    p.node(f'out{lane}',f'P{lane}\n→ ALIGN',1420,y+5,140,90,'port')
    p.edge('route',f'operand{lane}',points=[(590,y+50)],exit=(1,.2 if lane==0 else .85))
    p.edge(f'operand{lane}',f'mul{lane}')
    p.edge(f'mul{lane}',f'prod{lane}')
    p.edge(f'prod{lane}',f'out{lane}')
p.node('expin','E_A0 / E_B0',650,150,150,70,'port',16)
p.node('expadd','+',925,150,70,70,'op')
p.node('expreg','指数寄存器\nexp0：9bit',1100,135,190,100,'reg',16)
p.node('expout','exp0 → ALIGN',1420,150,170,70,'port',16)
p.edge('expin','expadd');p.edge('expadd','expreg');p.edge('expreg','expout')
p.node('expin1','E_A1 / E_B1',650,450,150,70,'port',16)
p.node('expadd1','+',925,450,70,70,'op')
p.node('expreg1','指数寄存器\nexp1：6bit',1100,435,190,100,'reg',16)
p.node('expout1','exp1 → ALIGN',1420,450,170,70,'port',16)
p.edge('expin1','expadd1');p.edge('expadd1','expreg1');p.edge('expreg1','expout1')
p.node('mode','mode / phase',310,160,210,65,'control',16)
p.edge('mode','route',exit=(.5,1),entry=(.5,0),control=True)
p.node('meta','nz · group_valid\nfirst · tag · scale · bias',650,790,320,80,'control',16)
p.node('metareg','metadata 寄存器',1100,790,220,80,'reg',16)
p.edge('meta','metareg',control=True)
p.text('phase_note','INT16：phase0 → LL / HL；phase1 → LH / HH',300,715,1100,45,16)
p.footer('FP19 使用 MUL12；FP9 / INT8 / INT4 使用两路独立乘法。INT16 重构由 PSUM 完成。',895)

# 3. Alignment. Identical lanes are shown independently to preserve group semantics.
p=Page(2,'ALIGN：分组对阶与逐项舍入','每组 MAX16；每个前端各有 lane0 / lane1 对阶通路',1810,1260)
p.root.find("./mxCell[@id='subtitle']/mxGeometry").set('height','20')
for lane,dy in [(0,0),(1,535)]:
    suf=str(lane)
    p.text('label'+suf,f'group{lane} / lane{lane}',50,140+dy,260,40,20,True)
    p.node('eall'+suf,'16 × exponent\n+ nz 掩码',50,220+dy,170,80,'port',16)
    p.node('max'+suf,'MAX16\n四级比较树',290,220+dy,190,80)
    p.node('minus'+suf,'−',600,225+dy,70,70,'op')
    p.node('ei'+suf,'E_i',592.5,145+dy,85,40,'port',16)
    p.edge('eall'+suf,'max'+suf);p.edge('max'+suf,'minus'+suf,label='Emax')
    p.edge('ei'+suf,'minus'+suf,exit=(.5,1),entry=(.5,0))
    p.node('pin'+suf,'P_i',50,365+dy,170,85,'port',18)
    lift='FP19：SEXT32 << 9\nFP9：SEXT24 << 15\nFP9 lane0 → SEXT32' if lane==0 else 'FP9：SEXT24 << 15'
    p.node('lift'+suf,lift,290,340+dy,240,140,size=16)
    p.node('shift'+suf,'算术右移 >>> d\nq + 丢弃位\nd ≥ W → 0',800,340+dy,200,140,size=16)
    p.edge('pin'+suf,'lift'+suf);p.edge('lift'+suf,'shift'+suf)
    p.edge('minus'+suf,'shift'+suf,points=[(900,260+dy)],entry=(.5,0),label='d')
    p.node('inc'+suf,'RNE 进位\nG ∧ (R ∨ S ∨ q₀)',780,530+dy,240,75,size=16)
    p.node('add'+suf,'+',1110,370+dy,80,80,'op')
    p.edge('shift'+suf,'add'+suf,label='q')
    p.edge('shift'+suf,'inc'+suf,exit=(.5,1),entry=(.5,0),label='GRS')
    p.edge('inc'+suf,'add'+suf,points=[(1150,567+dy)],entry=(.5,1),label='inc')
    p.node('mux'+suf,'模式\n选择',1290,350+dy,95,130,'mux')
    p.edge('add'+suf,'mux'+suf)
    p.edge('pin'+suf,'mux'+suf,points=[(135,490+dy),(245,490+dy),(245,640+dy),(1337,640+dy)],exit=(.5,1),entry=(.5,1),label='整数旁路')
    p.node('reg'+suf,'ALIGN 寄存器\n'+('signed32' if lane==0 else 'signed24')+'\nEmax / metadata',1470,340+dy,235,155,'reg',17)
    p.edge('mux'+suf,'reg'+suf)
    p.edge('max'+suf,'reg'+suf,points=[(385,135+dy),(1587,135+dy)],exit=(.5,0),entry=(.5,0),control=True)
p.footer('两组独立选取 Emax。FP9 两路均保留 21 位小数；FP19 拆片在 PSUM 内进行。',1210)

# 4. Partial sum trees and reconstruction; only final output is registered.
p=Page(3,'PSUM：双树归约与结果重构','两个物理 limb tree；双组输出和宽结果重构采用不同数据路径',2170,1020)
p.node('in','ALIGN\n16 × lane0\n16 × lane1',50,355,155,270,'port')
p.node('adapt','模式映射 / FP19 拆片\nL：unsigned23\nH：signed9\nDUAL：lane0 / lane1',270,355,250,270,size=16)
p.edge('in','adapt')
for lane,y in [(0,230),(1,690)]:
    p.node(f'tree{lane}',f'DW02_tree {lane}\n16 × signed24\n两行 carry-save',640,y,230,140,size=17)
    p.node(f'cpa{lane}','+',1040,y+30,80,80,'fusion')
    p.text(f'cpa_label{lane}','DW01_add\nsigned28',940,y+135,170,60,16)
    p.edge('adapt',f'tree{lane}',points=[(580,y+70)],exit=(1,.12 if lane==0 else .88))
    p.edge(f'tree{lane}',f'cpa{lane}',points=[(950,y+40),(950,y+55)],exit=(1,.28),entry=(0,.3),label='28')
    p.edge(f'tree{lane}',f'cpa{lane}',points=[(980,y+100),(980,y+85)],exit=(1,.72),entry=(0,.7),label='28')
p.node('hshift','H << 23：FP19\nH << 8：INT16',1240,670,215,90,size=17)
p.node('merge','+',1470,455,80,80,'fusion')
p.edge('cpa0','merge',points=[(1330,300),(1330,495)],entry=(0,.5),label='L')
p.edge('cpa1','hshift',label='H')
p.edge('hshift','merge',points=[(1510,715)],entry=(.5,1))
p.node('phase','INT16 phase1：<< 8\n其余：直通',1620,450,200,90,size=16)
p.edge('merge','phase',label='signed36')
p.node('mode','输出\n选择',1860,350,90,290,'mux',17)
p.edge('phase','mode')
p.edge('cpa0','mode',points=[(1160,185),(1905,185)],exit=(1,.3),entry=(.5,0),label='group0 · 28bit')
p.edge('cpa1','mode',points=[(1180,875),(1905,875)],exit=(1,.7),entry=(.5,1),label='group1 · 28bit')
p.node('reg','PSUM 输出寄存器\n56bit',2000,420,160,150,'reg',17)
p.edge('mode','reg')
p.text('dual','DUAL_OUTPUT：group0 / group1 分别保存',1280,130,650,40,17)
p.text('wide','WIDE_SUM：FP19 / INT16 先重构为 signed36',1390,350,720,40,17)
p.footer('两树内部为组合归约。DUAL_OUTPUT 不跨组相加；INT16 第二拍权重仅在 PSUM 施加一次。',945)

# 5. Exact mapping to the common coefficient / exponent representation.
p=Page(4,'MX：公共数值表示转换','每组独立转换；FP19 / FP9 / MXINT8 → signed36 Q2.34 + E[7:0]',1910,1040)
p.node('in','PSUM\n56bit',50,320,170,220,'port')
p.node('unpack','模式解包\nWIDE：signed36\nDUAL：2 × signed28',290,320,230,220,size=17)
p.edge('in','unpack')
p.node('coeff','系数精确重定标\nFP19：S\nFP9：S << 8\nMXINT8：S << 17\nINT：SEXT36',650,250,270,230,size=18)
p.edge('unpack','coeff',points=[(585,430),(585,365)])
p.node('ereg','MX 输出寄存器\nM：signed36\nE：unsigned8\nmode / first / tag',1530,280,230,460,'reg',20)
p.edge('coeff','ereg',points=[(1400,365)],entry=(0,.18),label='M')
p.node('emax','Emax',320,650,170,65,'port')
p.node('bias','bias_sum',540,850,190,65,'port',16)
p.node('minus','−',600,640,80,80,'op')
p.edge('emax','minus');p.edge('bias','minus',points=[(640,800)],exit=(.5,0),entry=(.5,1))
p.node('const','132',845,520,110,60,'port')
p.node('addconst','+',860,640,80,80,'op')
p.edge('minus','addconst');p.edge('const','addconst',exit=(.5,1),entry=(.5,0))
p.node('scale','s_A / s_B\nE8M0 scale',1000,880,180,80,'port',16)
p.node('scadd','+',1240,880,80,80,'op')
p.edge('scale','scadd')
p.node('addscale','+',1240,640,80,80,'op')
p.edge('addconst','addscale');p.edge('scadd','addscale',exit=(.5,0),entry=(.5,1),label='MX scale_sum')
p.node('expmux','E 选择\nINT：0',1380,615,120,130,'mux',16)
p.edge('addscale','expmux');p.edge('expmux','ereg',entry=(0,400/460),label='E')
p.node('output','→ ACC0 / ACC1',1800,460,105,100,'port',16)
p.edge('ereg','output')
p.text('scnote','普通浮点：scale_sum = 0；MXINT8：Emax = bias_sum = 0',700,775,800,40,16)
p.footer('重定标为精确补零 / 重解释；不执行舍入，不重复 PSUM 重构或 INT16 phase 权重。',985)

# 6. Independent accumulators. Floating and integer states below are logical views,
# not a prescription to duplicate physical register banks.
p=Page(5,'ACC：分组累加与状态反馈','单组通路示意；ACC0 / ACC1 独立运行。浮点与整数为互斥模式的逻辑视图。',1990,1530)
p.text('float-label','浮点 / MXINT8',50,140,400,40,21,True)
p.node('fnew','新贡献\nM[35:0] / E[7:0]',50,230,205,110,'port',17)
p.node('falign','MAX 指数 / 两路对阶\n保留低位 tail',580,210,235,160,size=18)
p.edge('fnew','falign')
p.node('fzero','0',140,490,95,60,'port')
p.node('fmux','first\n选零',350,450,120,150,'mux')
p.edge('fzero','fmux')
p.edge('fmux','falign',points=[(410,405),(697,405)],exit=(.5,0),entry=(.5,1),label='旧状态')
p.node('fadd','+',910,250,90,90,'op')
p.edge('falign','fadd',points=[(860,255)],exit=(1,.28),entry=(0,.28),label='q_new')
p.edge('falign','fadd',points=[(870,335)],exit=(1,.72),entry=(0,.72),label='q_old')
p.node('rne','规格化 / 最终 RNE\n37bit 和 + tail',1090,225,210,140,size=18)
p.edge('fadd','rne')
p.edge('falign','rne',points=[(790,420),(1195,420)],exit=(.9,1),entry=(.5,1),label='tail')
p.node('bound','写回范围处理\nFTZ / Inf 保持',1360,225,190,140,size=17)
p.edge('rne','bound')
p.node('fstate','累加状态寄存器\nM[35:0] / E[7:0]\nInf 状态',1610,205,210,190,'reg',17)
p.edge('bound','fstate')
p.node('fsnapshot','输出\n保持槽',1880,230,100,130,'reg',17)
p.edge('fstate','fsnapshot')
p.edge('fstate','fmux',points=[(1715,665),(410,665)],exit=(.5,1),entry=(.5,1))
p.text('tailnote','tail 的硬件工作位宽未冻结；此处保留功能边界。',650,540,890,40,16)
p.text('integer-label','整数（INT8 / INT4 / INT16）',50,830,620,40,21,True)
p.node('inew','新贡献\nsigned36',50,945,205,105,'port',17)
p.node('iadd','+',910,950,90,90,'op')
p.edge('inew','iadd')
p.node('izero','0',140,1240,95,60,'port')
p.node('imux','first\n选零',350,1190,120,150,'mux')
p.edge('izero','imux')
p.edge('imux','iadd',points=[(410,1140),(955,1140)],exit=(.5,0),entry=(.5,1),label='旧 INT32 / work36')
p.node('sat','INT32 饱和\nSEXT36',1090,945,210,110,size=18)
p.edge('iadd','sat')
p.node('phase','phase\n选择',1370,940,120,180,'mux')
p.edge('sat','phase',entry=(0,.5))
p.edge('iadd','phase',points=[(955,900),(1430,900)],exit=(.5,0),entry=(.5,0),label='INT16 phase0：不饱和')
p.node('istate','整数状态寄存器\nINT32 / work36',1610,945,210,170,'reg',18)
p.edge('phase','istate')
p.node('isnapshot','输出\n保持槽',1880,965,100,130,'reg',17)
p.edge('istate','isnapshot')
p.edge('istate','imux',points=[(1715,1410),(410,1410)],exit=(.5,1),entry=(.5,1))
p.text('commit','INT16：phase0 保存 work36；phase1 饱和并提交。INT8 / INT4 每次有效更新后饱和。',690,1245,1190,60,17)
p.footer('输出为内部累加快照；最终 FP32 转换在 MAC 外完成。寄存器及输出保持槽受 valid / ready 使能控制。',1470)

# Structural checks before writing any deliverable.
for diagram in doc.findall('diagram'):
    cells=diagram.findall('./mxGraphModel/root/mxCell')
    ids={c.get('id') for c in cells}
    assert len(ids)==len(cells)
    for c in cells:
        if c.get('edge')=='1':
            assert c.get('source') in ids and c.get('target') in ids
            assert 'endArrow=block' in c.get('style','')
            assert c.find('mxGeometry').get('relative')=='1'
        if c.get('value') in ['+','×','−']:
            assert 'shape=ellipse' in c.get('style','')
            g=c.find('mxGeometry');assert g.get('width')==g.get('height')
        if ('寄存器' in c.get('value','') or c.get('value')=='输出\n保持槽') and 'text;' not in c.get('style',''):
            assert 'rounded=1' in c.get('style','')
ET.indent(doc,space='  ')
ET.ElementTree(doc).write(TARGET,encoding='utf-8',xml_declaration=True)
for i,d in enumerate(doc.findall('diagram'),1):
    one=ET.Element('mxfile',host='Electron',pages='1')
    one.append(copy.deepcopy(d))
    ET.ElementTree(one).write(OUT/f'{i:02d}.drawio',encoding='utf-8',xml_declaration=True)
print(f'Written {TARGET}: six editable pages. Backup: {BACKUP}')
