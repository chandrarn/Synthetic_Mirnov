# Contribution of W, He3 to Zeff and quasineutrality
Zeff_imp = (
    (
        inga.profiles['z'][iW]**2 
        * inga.profiles['ni(10^19/m^3)'][:,iW]/inga.profiles['ne(10^19/m^3)']
        )
    + (
        inga.profiles['z'][iHe3]**2 
        * inga.profiles['ni(10^19/m^3)'][:,iHe3]/inga.profiles['ne(10^19/m^3)']
        )
    ) # dim(nrho,)
Zeff_star = Zeff - Zeff_imp
QN_star = 1- (
    (
        inga.profiles['z'][iW]
        * inga.profiles['ni(10^19/m^3)'][:,iW]/inga.profiles['ne(10^19/m^3)']
        )
    + (
        inga.profiles['z'][iHe3]
        * inga.profiles['ni(10^19/m^3)'][:,iHe3]/inga.profiles['ne(10^19/m^3)']
        )
    )


# Calculates new dilution
c_L = (
    (Zeff_star - QN_star)
    /(Z_L**2 - Z_L)
    )
c_D = (
    (QN_star - Z_L*c_L)
    /(1+ToD)
    )
c_T = ToD*c_D