"""sweep 로그(p1/p2/p3)의 숫자를 읽어 영어 라벨 그래프로 재플롯(재실행 없음)."""
import re, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
SCR="/tmp/claude-0/-home-user-tmp-ML/9886526d-b0a7-5204-b0e9-6828e109e84f/scratchpad"
BUD=[180,780,2400]
def parse(path):
    d={}
    for ln in open(path):
        m=re.match(r"\s+(\S+)\s+B=180:",ln)
        if not m: continue
        nums=re.findall(r"B=\d+:\s*([\d.]+)",ln)
        if len(nums)==3: d[m.group(1)]=[float(x) for x in nums]
    return d
def plot(d,title,out,ylabel):
    plt.figure(figsize=(9,5.5))
    for n,v in d.items(): plt.plot(BUD,[max(x,1e-6) for x in v],marker="o",label=n,lw=1.7)
    plt.xscale("log");plt.yscale("log");plt.xticks(BUD,[str(b) for b in BUD])
    plt.xlabel("iter (evaluation) budget");plt.ylabel(ylabel)
    plt.title(f"{title}\niter budget sweep (lower=better)")
    plt.grid(alpha=0.3,which="both");plt.legend(fontsize=9,ncol=2)
    plt.tight_layout();plt.savefig(out,dpi=120);print("saved",out)
plot(parse(f"{SCR}/p1.log"),"prob1 — iter budget","prob1/itersweep.png","gap to optimum (%, mean 3 cases)")
plot(parse(f"{SCR}/p2.log"),"prob2 block — iter budget","prob2/itersweep.png","normalized MSE")
plot(parse(f"{SCR}/p3.log"),"prob3 TRAP — iter budget","prob3/itersweep.png","normalized MSE")
