import numpy as np
p = "/Users/xiaxin/work/WindForecast_Project/03_Results/re-plot-figures/figure-2/correlations_all.npz"
d = np.load(p, allow_pickle=True)

for k in ["10m-70m-free","10m-70m-wake","30m-70m-free","30m-70m-wake",
          "10m-power-free","10m-power-wake","70m-power-free","70m-power-wake"]:
    T, R = d[k+"_T"], d[k+"_R"]
    print("\n"+k)
    for i,(t,r) in enumerate(zip(T,R), 1):
        print(f"  IMF{i}  T={t:8.2f} h   R={r:6.3f}")