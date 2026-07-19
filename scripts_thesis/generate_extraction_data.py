"""Generate output/all76.json with all 76 extracted RT2-SOA conditions,
corrected values, and inclusion flags. Run once; figure scripts read it."""
import json, os

OUT = os.path.join(os.path.dirname(__file__), '..', 'output', 'all76.json')

D = []
def add(study, exp, cond, soas, rt2, rt1, src, flags=()):
    D.append(dict(study=study, exp=exp, cond=cond, soas=soas, rt2=rt2,
                  rt1=rt1, src=src, flags=list(flags)))

# ── McCann & Johnston 1992 ──
S=[50,150,300,800]
add("McCann & Johnston (1992)","1","Arbitrary mapping",S,[860,805,740,710],455,"F")
add("McCann & Johnston (1992)","1","Ordered mapping",S,[805,730,695,630],455,"F")
add("McCann & Johnston (1992)","2","Overall",S,[722,656,594,524],490,"T")

# ── Pashler & Johnston 1989 ──
S=[50,100,400]
add("Pashler & Johnston (1989)","1","Standard",S,[859,806,621],593,"T")
add("Pashler & Johnston (1989)","2","Grouped",S,[825,789,632],866,"T",['grouped'])

# ── Osman & Moore 1993 ──
add("Osman & Moore (1993)","1","Overall",[50,200,500],[787,631,508],539,"T")

# ── Pashler 1990 ──
S=[100,200,700]
add("Pashler (1990)","1","Tone first, known",S,[820,740,540],750,"F")
add("Pashler (1990)","1","Letter first, known",S,[850,750,550],750,"F")
add("Pashler (1990)","1","Tone first, unknown",S,[1025,965,620],750,"F",['unknown_order'])
add("Pashler (1990)","1","Letter first, unknown",S,[980,960,930],750,"F",['unknown_order'])
add("Pashler (1990)","2","Tone first, known",S,[730,680,550],700,"F")
add("Pashler (1990)","2","Tone first, unknown",S,[710,720,575],625,"F",['unknown_order'])
add("Pashler (1990)","2","Letter first, known",S,[800,740,600],710,"F")
add("Pashler (1990)","2","Letter first, unknown",S,[780,770,670],575,"F",['unknown_order'])
add("Pashler (1990)","3","Known order",S,[610,530,420],490,"F")
add("Pashler (1990)","3","Unknown, flashes first",S,[775,675,510],680,"F",['unknown_order'])
add("Pashler (1990)","3","Unknown, digit first",S,[825,710,500],670,"F",['unknown_order'])

# ── De Jong 1993 ──
S1=[50,250,350,450,700]; S2=[25,250,800]
add("De Jong (1993)","1","Go, Choice, Dim",S1,[610,500,460,420,410],425,"F")
add("De Jong (1993)","1","Go, Choice, Bright",S1,[610,480,440,410,390],425,"F")
add("De Jong (1993)","1","Nogo, Choice, Dim",S1,[525,460,430,440,420],425,"F",['nogo'])
add("De Jong (1993)","1","Nogo, Choice, Bright",S1,[525,445,415,400,390],425,"F",['nogo'])
add("De Jong (1993)","2","Hand",S2,[560,430,380],380,"F")
add("De Jong (1993)","2","Foot",S2,[640,520,470],380,"F")
add("De Jong (1993)","3","Go, Uppercase",S2,[620,470,400],400,"F")
add("De Jong (1993)","3","Go, Lowercase",S2,[630,500,440],400,"F")
add("De Jong (1993)","3","Nogo, Uppercase",S2,[560,460,410],400,"F",['nogo'])
add("De Jong (1993)","3","Nogo, Lowercase",S2,[590,500,450],400,"F",['nogo'])
add("De Jong (1993)","4","Separate, Dim, Lowercase",S2,[660,510,460],400,"F")
add("De Jong (1993)","4","Separate, Dim, Uppercase",S2,[660,500,410],400,"F")
add("De Jong (1993)","4","Separate, Bright, Lowercase",S2,[660,510,440],400,"F")
add("De Jong (1993)","4","Separate, Bright, Uppercase",S2,[660,500,390],400,"F")
add("De Jong (1993)","4","Grouped, Dim, Lowercase",S2,[640,510,480],400,"F",['grouped'])
add("De Jong (1993)","4","Grouped, Dim, Uppercase",S2,[600,500,440],400,"F",['grouped'])
add("De Jong (1993)","4","Grouped, Bright, Lowercase",S2,[625,500,450],400,"F",['grouped'])
add("De Jong (1993)","4","Grouped, Bright, Uppercase",S2,[600,480,410],400,"F",['grouped'])
add("De Jong (1993)","5","Go, Same, Name",S2,[660,590,515],360,"F")
add("De Jong (1993)","5","Go, Diff, Name",S2,[660,590,515],360,"F")
add("De Jong (1993)","5","Go, Same, Physical",S2,[590,470,420],360,"F")
add("De Jong (1993)","5","Go, Diff, Physical",S2,[625,500,480],360,"F")
add("De Jong (1993)","5","Go, Simple",S2,[500,350,210],360,"F",['simple_rt'])
add("De Jong (1993)","5","Nogo, Same, Name",S2,[650,590,550],360,"F",['nogo'])
add("De Jong (1993)","5","Nogo, Diff, Name",S2,[660,590,550],360,"F",['nogo'])
add("De Jong (1993)","5","Nogo, Same, Physical",S2,[540,460,410],360,"F",['nogo'])
add("De Jong (1993)","5","Nogo, Diff, Physical",S2,[590,500,470],360,"F",['nogo'])
add("De Jong (1993)","5","Nogo, Simple",S2,[410,280,210],360,"F",['simple_rt','nogo'])

# ── Van Selst 1999 ──
S=[17,67,150,250,450,850]
add("Van Selst et al. (1999)","1","Session 1",S,[1125,1025,1020,900,820,800],600,"F")
add("Van Selst et al. (1999)","1","Session 18",S,[540,510,490,480,490,500],310,"F",['practice_phase'])
add("Van Selst et al. (1999)","1","Session 26",S,[500,480,460,445,465,460],310,"F",['practice_phase'])

# ── Schubert 1999 ──
S=[50,100,350,800]
add("Schubert (1999)","1","One-alt (overall)",S,[595,560,400,250],475,"F",['simple_rt'])
add("Schubert (1999)","1","Two-alt (overall)",S,[710,660,510,425],540,"F")
add("Schubert (1999)","1","Two-alt, 1-2 order",S,[700,660,500,450],540,"F",['suborder'])
add("Schubert (1999)","1","Two-alt, 2-1 order",S,[710,650,480,440],540,"F",['suborder'])
add("Schubert (1999)","1","One-alt, 1-2 order",S,[595,560,400,220],475,"F",['simple_rt','suborder'])
add("Schubert (1999)","1","One-alt, 2-1 order",S,[595,560,400,280],475,"F",['simple_rt','suborder'])
add("Schubert (1999)","2","Two-alt (p=0.5)",S,[925,875,650,525],725,"F")
add("Schubert (1999)","2","One-alt (p=0.5)",S,[775,700,500,490],610,"F",['simple_rt'])
add("Schubert (1999)","3","Three-alt",S,[1025,990,780,610],800,"F")
add("Schubert (1999)","3","Two-alt",S,[975,925,700,550],800,"F")
add("Schubert (1999)","3","One-alt (p=0.5)",S,[805,775,575,550],600,"F",['simple_rt'])

# ── Lien et al. 2005 ──
S=[0,50,150,300,500,1000]
add("Lien et al. (2005)","1","nonIM-nonIM, Noncorr",S,[915,910,840,795,750,715],556,"F")
add("Lien et al. (2005)","1","nonIM-nonIM, Corr",S,[895,890,805,760,750,715],556,"F")
add("Lien et al. (2005)","2","IM-nonIM, Noncorr",S,[780,750,720,710,700,680],455,"F")
add("Lien et al. (2005)","2","IM-nonIM, Corr",S,[735,720,700,690,660,650],455,"F")
add("Lien et al. (2005)","3","nonIM-IM, Noncorr",S,[720,695,670,660,640,600],499,"F")
add("Lien et al. (2005)","3","nonIM-IM, Corr",S,[705,685,660,640,630,620],499,"F")
add("Lien et al. (2005)","4","IM-IM, Noncorr",S,[690,670,660,650,640,630],422,"F")
add("Lien et al. (2005)","4","IM-IM, Corr",S,[685,670,645,650,640,625],422,"F")

# ── Sigman & Dehaene 2008 ──
S=[0,300,900,1200]
add("Sigman & Dehaene (2008)","fMRI","Overall",S,[1053,784,543,490],618,"T")
add("Sigman & Dehaene (2008)","EEG","Overall",S,[990,742,520,480],622,"T")

# ── Halvorson et al. 2013 ──
add("Halvorson et al. (2013)","1","PRP blocks (IM-IM)",[0,200,800],[667,573,498],476,"T")

# ── Rau & Zheng 2020 ──
S=[75,150,300,600,1200]
add("Rau & Zheng (2020)","1","Aud-Tact",S,[1540,1490,1230,900,790],1010,"T")
add("Rau & Zheng (2020)","1","Aud-Vis",S,[1250,1190,1000,770,660],1010,"T")
add("Rau & Zheng (2020)","1","Tact-Vis",S,[1110,990,800,670,590],830,"T")
add("Rau & Zheng (2020)","1","Tact-Aud",S,[1270,1230,1070,970,790],850,"T")
add("Rau & Zheng (2020)","1","Vis-Aud",S,[1070,980,900,810,760],730,"T")
add("Rau & Zheng (2020)","1","Vis-Tact",S,[1120,1040,860,760,670],820,"T")

# ── Compute derived fields ──
for d in D:
    soas, rt2 = d['soas'], d['rt2']
    d['head'] = (rt2[1] - rt2[0]) / (soas[1] - soas[0])
    d['soa_star'] = 0.80 * d['rt1'] if d['rt1'] else None
    d['adj_slopes'] = [round((rt2[i+1] - rt2[i]) / (soas[i+1] - soas[i]), 2)
                       for i in range(len(soas) - 1)]
    ss = d['soa_star']
    if ss and len(soas) >= 2 and soas[-2] >= ss:
        d['tail'] = (rt2[-1] - rt2[-2]) / (soas[-1] - soas[-2])
        d['tail_clean'] = True
    else:
        d['tail'] = ((rt2[-1] - rt2[-2]) / (soas[-1] - soas[-2])
                     if len(soas) >= 2 else None)
        d['tail_clean'] = False
    # Van Selst Session 1 tail demotion: 450ms < SOA*=480ms
    if d['study'].startswith('Van Selst') and d['cond'] == 'Session 1':
        d['tail_clean'] = False

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(D, open(OUT, 'w'), indent=1)
print(f"Written {len(D)} conditions to {OUT}")