function create_metrics(pm, Nr, Ns,
                        xf = (r,s)->(r,ones(size(r)),zeros(size(r)))
                        yf = (r,s)->(s,zeros(size(s)),ones(size(s))))
    Nrp = Nr + 1
    Nsp = Ns + 1
    Np = Nrp + Nsp
end