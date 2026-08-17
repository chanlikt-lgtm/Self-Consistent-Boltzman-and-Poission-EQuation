(* ::Package:: *)
(* ================================================================
   liang1997_she.wls  (v2, clean)
   Mathematica port of the Liang, Goldsman, Mayergoyz & Oldiges (1997)
   IEEE TED 44(2) deterministic first-order SHE-BTE 2-D MOSFET
   reproduction (Python: reproduce_all.py + src/*.py).

   Sections:
     S1 constants            (src/constants.py)
     S2 band structure       (src/bands.py)
     S3 device + mesh        (src/device.py)
     S4 2-D nonlinear Poisson(src/poisson.py)
     S5 drift-diffusion      (src/dd.py)
     S6 bulk SHE validation  (src/she_bulk.py)
     S7 2-D SHE-BTE (r,H)    (src/she2d.py, run_she_clean.py)
     S8 coupled loop demo    (src/coupled_she.py)
     S9 verification vs Python reference (data/*.npz -> ref/*.csv)

   NOTE on sparse assembly: scipy.sparse SUMS duplicate (i,j) entries;
   Mathematica SparseArray lets the LAST rule win.  All diagonal
   contributions are therefore accumulated in separate per-pattern
   SparseArrays and summed with Total[].
   ================================================================ *)
$HistoryLength = 0;
dir = DirectoryName[$InputFileName];
SetDirectory[dir];
Print["Mathematica ", $Version];
Print["Working dir: ", dir];

(* =====[SECTION: S1 constants]===== *)
q    = 1.602176634*^-19;      (* C *)
eps0 = 8.8541878128*^-14;     (* F/cm *)
kB   = 1.380649*^-23;         (* J/K *)
hbar = 1.054571817*^-34;      (* J s *)
h    = 6.62607015*^-34;       (* J s *)
m0   = 9.1093837015*^-31;     (* kg *)
eV   = 1.602176634*^-19;      (* J *)
T    = 300.0;                 (* K *)
Vt   = kB*T/q;
epsSi = 11.7; epsOx = 3.9;
ni   = 1.45*^10;              (* cm^-3 *)
mstar = 0.32*m0;
Dac  = 4.0; Dn = 5.0*^8; Eop = 0.05; vsat = 9.0*^6; GammaSR = 1.96;
nscr = 1.45*^15; Dsa = 17.8; deltaQ = 13.1; epsMaxEV = 3.02;
rhoSi = 2.329; uSound = 9.04*^5;
KT   = kB*T;
EPSMAX = epsMaxEV*eV;

(* =====[SECTION: S2 bands]===== *)
MSTAR = 0.32*m0; ALPHA = 0.5; GSV = 12;
gammaFn[eps_]  := eps*(1 + ALPHA*eps);
gammaPFn[eps_] := 1 + 2*ALPHA*eps;
velFn[eps_]    := Sqrt[2*Max[gammaFn[eps]*eV, 0]/MSTAR]/gammaPFn[eps];          (* m/s *)
dosFn[eps_]    := GSV/(2 Pi^2)*(2 MSTAR)^1.5/(2 hbar^3)*Sqrt[Max[gammaFn[eps]*eV, 0]]*gammaPFn[eps]; (* 1/(J m^3) *)
dosPerEV[eps_] := dosFn[eps]*eV;
trap1D[f_, x_] := Total[0.5 (f[[2 ;;]] + f[[1 ;; -2]])*Differences[x]];
epsGrid = Range[1*^-5, 3.02, (3.02 - 1*^-5)/3999.0];
NcNP = trap1D[Table[dosPerEV[e]*1*^-6*Exp[-e*eV/(kB T)], {e, epsGrid}], epsGrid];
NcParAn = 2*(GSV/2)*(2 Pi*MSTAR*kB*T/h^2)^1.5*1*^-6;
Print["S2 bands: v(0.1 eV) = ", ScientificForm[velFn[0.1]*100, 4], " cm/s,  v(1 eV) = ", ScientificForm[velFn[1.0]*100, 4], " cm/s"];
Print["S2 bands: Nc(nonpar) = ", ScientificForm[NcNP, 4], " cm^-3,  Nc(par,analytic) = ", ScientificForm[NcParAn, 4], " cm^-3"];

(* =====[SECTION: S3 device]===== *)
dev = <|"Lx" -> 0.80, "Ly" -> 0.70, "tox" -> 0.0096,
        "x_gate1" -> 0.20, "x_gate2" -> 0.60,
        "x_srcnplus" -> 0.150, "x_drnnplus" -> 0.650,
        "x_srcldd" -> 0.220, "x_drnldd" -> 0.580,
        "xj_nplus" -> 0.105, "xj_ldd" -> 0.060,
        "sig_lat" -> 0.028, "sig_dep" -> 0.025,
        "x_chimp1" -> 0.18, "x_chimp2" -> 0.62,
        "xj_chimp" -> 0.060, "sig_chimp_dep" -> 0.035,
        "Nd_nplus" -> 1.0*^20, "Nd_ldd" -> 1.0*^18,
        "Na_sub" -> 1.0*^16, "Na_chimp" -> 4.0*^17|>;
latL[x_, edge_, sig_] := 0.5 Erfc[(x - edge)/sig];
latR[x_, edge_, sig_] := 0.5 Erfc[(edge - x)/sig];
depthF[y_, xj_, sig_] := 0.5 Erfc[(y - xj)/sig];
makeMesh[Nx_, Ny_] := Module[{x, s, y, X, Y},
  x = Range[0.0, dev["Lx"], dev["Lx"]/(Nx - 1)];
  s = Range[0.0, 1.0, 1.0/(Ny - 1)];
  y = dev["Ly"]*(s^1.8);
  X = ConstantArray[x, Ny];
  Y = Transpose[ConstantArray[y, Nx]];
  {x, y, X, Y}];
buildDoping[{X_, Y_}] := Module[{p = dev, srcN, drnN, srcL, drnL, Nd, sub, chimp, Na, Nnet},
  srcN = p["Nd_nplus"]*latL[X, p["x_srcnplus"], p["sig_lat"]]*depthF[Y, p["xj_nplus"], p["sig_dep"]];
  drnN = p["Nd_nplus"]*latR[X, p["x_drnnplus"], p["sig_lat"]]*depthF[Y, p["xj_nplus"], p["sig_dep"]];
  srcL = p["Nd_ldd"]*latL[X, p["x_srcldd"], p["sig_lat"]]*depthF[Y, p["xj_ldd"], p["sig_dep"]];
  drnL = p["Nd_ldd"]*latR[X, p["x_drnldd"], p["sig_lat"]]*depthF[Y, p["xj_ldd"], p["sig_dep"]];
  Nd = srcN + drnN + srcL + drnL;
  sub = p["Na_sub"]*ConstantArray[1.0, Dimensions[X]];
  chimp = p["Na_chimp"]*(0.5 (Erfc[(X - p["x_chimp2"])/p["sig_lat"]] - Erfc[(X - p["x_chimp1"])/p["sig_lat"]]))*depthF[Y, p["xj_chimp"], p["sig_chimp_dep"]];
  Na = sub + chimp;
  Nnet = Nd - Na;
  {Nd, Na, Nnet}];

(* =====[SECTION: S4 poisson]===== *)
UM = 1*^-4;                    (* cm per um *)
poissonInit[x_um_, y_um_, Nnet_, PhiGate_: 0.30] := Module[
  {x, y, Nx, Ny, N, dxc, dyc, dxe, dyn, vol, e0si, e0ox,
   dirichlet, contactType, gateMask, phiEq, rows, cols, vals, diagGeom, bGate, L, i, j, k, Tf, src, drn},
  x = x_um*UM; y = y_um*UM;
  Nx = Length[x]; Ny = Length[y]; N = Nx*Ny;
  dxc = Table[If[i == 1, 0.5 (x[[2]] - x[[1]]), If[i == Nx, 0.5 (x[[Nx]] - x[[Nx - 1]]), 0.5 (x[[i + 1]] - x[[i - 1]])]], {i, 1, Nx}];
  dyc = Table[If[j == 1, 0.5 (y[[2]] - y[[1]]), If[j == Ny, 0.5 (y[[Ny]] - y[[Ny - 1]]), 0.5 (y[[j + 1]] - y[[j - 1]])]], {j, 1, Ny}];
  dxe = Differences[x]; dyn = Differences[y];
  vol = Outer[Times, dyc, dxc];
  e0si = epsSi*eps0; e0ox = epsOx*eps0;
  dirichlet = ConstantArray[False, {Ny, Nx}];
  contactType = ConstantArray[0, {Ny, Nx}];
  Do[If[x_um[[i]] <= dev["x_srcnplus"], dirichlet[[1, i]] = True; contactType[[1, i]] = 1], {i, 1, Nx}];
  Do[If[x_um[[i]] >= dev["x_drnnplus"], dirichlet[[1, i]] = True; contactType[[1, i]] = 2], {i, 1, Nx}];
  dirichlet[[Ny, All]] = True; contactType[[Ny, All]] = 3;
  gateMask = ConstantArray[False, {Ny, Nx}];
  Do[If[(x_um[[i]] >= dev["x_gate1"]) && (x_um[[i]] <= dev["x_gate2"]) && (! dirichlet[[1, i]]), gateMask[[1, i]] = True], {i, 1, Nx}];
  phiEq = Vt*ArcSinh[Nnet/(2 ni)];
  idx[j_, i_] := (j - 1) Nx + i;
  rows = {}; cols = {}; vals = {};
  diagGeom = ConstantArray[0.0, N];
  Do[
    k = idx[j, i];
    If[i + 1 <= Nx, Tf = e0si*dyc[[j]]/dxe[[i]];
      AppendTo[rows, k]; AppendTo[cols, idx[j, i + 1]]; AppendTo[vals, -Tf]; diagGeom[[k]] += Tf];
    If[i - 1 >= 1, Tf = e0si*dyc[[j]]/dxe[[i - 1]];
      AppendTo[rows, k]; AppendTo[cols, idx[j, i - 1]]; AppendTo[vals, -Tf]; diagGeom[[k]] += Tf];
    If[j + 1 <= Ny, Tf = e0si*dxc[[i]]/dyn[[j]];
      AppendTo[rows, k]; AppendTo[cols, idx[j + 1, i]]; AppendTo[vals, -Tf]; diagGeom[[k]] += Tf];
    If[j - 1 >= 1, Tf = e0si*dxc[[i]]/dyn[[j - 1]];
      AppendTo[rows, k]; AppendTo[cols, idx[j - 1, i]]; AppendTo[vals, -Tf]; diagGeom[[k]] += Tf],
    {j, 1, Ny}, {i, 1, Nx}];
  Do[AppendTo[rows, k]; AppendTo[cols, k]; AppendTo[vals, diagGeom[[k]]], {k, 1, N}];
  (* gate Robin *)
  bGate = ConstantArray[0.0, N];
  Do[If[gateMask[[1, i]], k = idx[1, i];
     Tf = e0ox*dxc[[i]]/(dev["tox"]*UM);
     diagGeom[[k]] += Tf; bGate[[k]] += Tf], {i, 1, Nx}];
  Do[AppendTo[rows, k]; AppendTo[cols, k]; AppendTo[vals, diagGeom[[k]]], {k, 1, N}];
  L = SparseArray[Transpose[{rows, cols}] -> vals, {N, N}];
  <|"x" -> x, "y" -> y, "Nx" -> Nx, "Ny" -> Ny, "N" -> N, "Nnet" -> Nnet,
    "dxc" -> dxc, "dyc" -> dyc, "dxe" -> dxe, "dyn" -> dyn, "vol" -> vol,
    "L" -> L, "bGate" -> bGate, "dirichlet" -> dirichlet,
    "contactType" -> contactType, "gateMask" -> gateMask, "phiEq" -> phiEq,
    "PhiGate" -> PhiGate|>];
carriers[phi_, phiN_, phiP_] := Module[{cl = 80.0},
  {ni*Exp[Clip[(phi - phiN)/Vt, {-cl, cl}]], ni*Exp[Clip[(phiP - phi)/Vt, {-cl, cl}]]}];
poissonSolve[ps_, Vs_: 0, Vd_: 0, Vg_: 0, Vb_: 0, phiInit_: None, phiN_: None, phiP_: None,
  maxNewton_: 60, tol_: 1*^-8, dampClip_: 0.5] := Module[
  {Nx, Ny, N, NnetF, volF, phiNF, phiPF, phiBC, dirF, phiBCF, phi, bGateV,
   n, pp, charge, F, jacDiag, J, dphi, res, it, dirPos},
  Nx = ps["Nx"]; Ny = ps["Ny"]; N = ps["N"];
  NnetF = Flatten[ps["Nnet"]]; volF = Flatten[ps["vol"]];
  phiNF = Flatten[If[phiN === None, ConstantArray[0.0, {Ny, Nx}], phiN]];
  phiPF = Flatten[If[phiP === None, ConstantArray[0.0, {Ny, Nx}], phiP]];
  phiBC = ps["phiEq"] + 0.0;
  phiBC[[Flatten[Position[ps["contactType"], 1]]]] += Vs;
  phiBC[[Flatten[Position[ps["contactType"], 2]]]] += Vd;
  phiBC[[Flatten[Position[ps["contactType"], 3]]]] += Vb;
  dirF = Flatten[ps["dirichlet"]];
  phiBCF = Flatten[phiBC];
  phi = If[phiInit === None, Flatten[ps["phiEq"]], Flatten[phiInit]];
  dirPos = Flatten[Position[dirF, True]];
  phi[[dirPos]] = phiBCF[[dirPos]];
  bGateV = ps["bGate"]*(Vg - ps["PhiGate"]);
  L = ps["L"];
  it = 0;
  Do[
    {n, pp} = carriers[ArrayReshape[phi, {Ny, Nx}], ArrayReshape[phiNF, {Ny, Nx}], ArrayReshape[phiPF, {Ny, Nx}]];
    n = Flatten[n]; pp = Flatten[pp];
    charge = q*(pp - n + NnetF)*volF;
    F = L.phi - bGateV - charge;
    jacDiag = q*(pp + n)/Vt*volF;
    J = L + SparseArray[Range[N] -> jacDiag];
    F[[dirPos]] = phi[[dirPos]] - phiBCF[[dirPos]];
    Do[J[[k, All]] = 0.0; J[[k, k]] = 1.0, {k, dirPos}];
    dphi = LinearSolve[J, -F];
    dphi = Clip[dphi, {-dampClip, dampClip}];
    phi += dphi;
    res = Max[Abs[dphi]];
    If[res < tol, Break[]],
    {it, 1, maxNewton}];
  phi2 = ArrayReshape[phi, {Ny, Nx}];
  {n2, p2} = carriers[phi2, ArrayReshape[phiNF, {Ny, Nx}], ArrayReshape[phiPF, {Ny, Nx}]];
  <|"phi" -> phi2, "n" -> n2, "p" -> p2, "iter" -> it, "res" -> res|>];

(* =====[SECTION: S5 dd]===== *)
bernV[x_] := Module[{sm, bg},
  sm = UnitStep[1*^-10 - Abs[x]]; bg = UnitStep[x - 40.0];
  (1 - sm) (1 - bg) (x/ExpM1[x]) + sm (1 - x/2) + bg (x Exp[-x])];
mobLow[Nabs_, car_] := Module[{muMin, muMax, Nref, a},
  If[car == "n", {muMin, muMax, Nref, a} = {68.5, 1414.0, 9.2*^16, 0.711},
   {muMin, muMax, Nref, a} = {44.9, 470.5, 2.23*^17, 0.719}];
  muMin + (muMax - muMin)/(1 + (Nabs/Nref)^a)];
mobField[muLF_, Efield_, car_] := Module[{vsF, beta},
  If[car == "n", {vsF, beta} = {1.07*^7, 2.0}, {vsF, beta} = {8.37*^6, 1.0}];
  muLF/(1 + (muLF*Abs[Efield]/vsF)^beta)^(1/beta)];
ddInit[ps_] := Module[{Nabs, neq},
  Nabs = Abs[ps["Nnet"]];
  neq = 0.5 (ps["Nnet"] + Sqrt[ps["Nnet"]^2 + 4 ni^2]);
  <|"ps" -> ps, "Nx" -> ps["Nx"], "Ny" -> ps["Ny"], "N" -> ps["N"],
    "munLF" -> mobLow[Nabs, "n"], "mupLF" -> mobLow[Nabs, "p"],
    "neq" -> neq, "peq" -> ni^2/neq, "dirichlet" -> ps["dirichlet"],
    "tauN" -> 1*^-7, "tauP" -> 1*^-7|>];
ddAssemble[dd_, phi_, n_, p_, car_, gen_: None] := Module[
  {Nx, Ny, N, muLF, Ex, Ey, mux, muy, rows, cols, vals, diag, b, Dsrh, vol,
   k, V, generation, dirPos, faces, fc, jj, ii, mu, hh, Af, C, dphi, Dv, j, i},
  Nx = dd["Nx"]; Ny = dd["Ny"]; N = dd["N"];
  muLF = If[car == "n", dd["munLF"], dd["mupLF"]];
  Ex = Table[Abs[phi[[j, i + 1]] - phi[[j, i]]]/dd["ps"]["dxe"][[i]], {j, 1, Ny}, {i, 1, Nx - 1}];
  mux = mobField[0.5 (muLF[[All, 1 ;; -2]] + muLF[[All, 2 ;;]]), Ex, car];
  Ey = Table[Abs[phi[[j + 1, i]] - phi[[j, i]]]/dd["ps"]["dyn"][[j]], {j, 1, Ny - 1}, {i, 1, Nx}];
  muy = mobField[0.5 (muLF[[1 ;; -2, All]] + muLF[[2 ;;, All]]), Ey, car];
  rows = {}; cols = {}; vals = {};
  diag = ConstantArray[0.0, N]; b = ConstantArray[0.0, N];
  Dsrh = dd["tauP"]*(n + ni) + dd["tauN"]*(p + ni);
  vol = dd["ps"]["vol"];
  generation = If[gen === None, ConstantArray[0.0, {Ny, Nx}], gen];
  idx[j_, i_] := (j - 1) Nx + i;
  Do[If[dd["dirichlet"][[j, i]], Continue[]];
    k = idx[j, i];
    faces = {};
    If[i + 1 <= Nx, AppendTo[faces, {j, i + 1, mux[[j, i]], dd["ps"]["dxe"][[i]], dd["ps"]["dyc"][[j]]}]];
    If[i - 1 >= 1, AppendTo[faces, {j, i - 1, mux[[j, i - 1]], dd["ps"]["dxe"][[i - 1]], dd["ps"]["dyc"][[j]]}]];
    If[j + 1 <= Ny, AppendTo[faces, {j + 1, i, muy[[j, i]], dd["ps"]["dyn"][[j]], dd["ps"]["dxc"][[i]]}]];
    If[j - 1 >= 1, AppendTo[faces, {j - 1, i, muy[[j - 1, i]], dd["ps"]["dyn"][[j - 1]], dd["ps"]["dxc"][[i]]}]];
    Do[
      {jj, ii, mu, hh, Af} = fc;
      C = q*mu*Vt*Af/hh;
      dphi = phi[[jj, ii]] - phi[[j, i]];
      Dv = dphi/Vt;
      If[car == "n",
        diag[[k]] += -C*bernV[-Dv]; AppendTo[rows, k]; AppendTo[cols, idx[jj, ii]]; AppendTo[vals, C*bernV[Dv]],
        diag[[k]] += -C*bernV[Dv]; AppendTo[rows, k]; AppendTo[cols, idx[jj, ii]]; AppendTo[vals, C*bernV[-Dv]]],
      {fc, faces}];
    V = vol[[j, i]];
    If[car == "n",
      diag[[k]] += -q*V*p[[j, i]]/Dsrh[[j, i]]; b[[k]] += -q*V*ni^2/Dsrh[[j, i]],
      diag[[k]] += -q*V*n[[j, i]]/Dsrh[[j, i]]; b[[k]] += -q*V*ni^2/Dsrh[[j, i]]];
    b[[k]] += -q*V*generation[[j, i]],
    {j, 1, Ny}, {i, 1, Nx}];
  Do[AppendTo[rows, k]; AppendTo[cols, k]; AppendTo[vals, diag[[k]]], {k, 1, N}];
  dirPos = Flatten[Position[Flatten[dd["dirichlet"]], True]];
  Do[k = dirPos[[m]];
    AppendTo[rows, k]; AppendTo[cols, k]; AppendTo[vals, 1.0];
    b[[k]] = If[car == "n", Flatten[dd["neq"]][[k]], Flatten[dd["peq"]][[k]]], {m, 1, Length[dirPos]}];
  {SparseArray[Transpose[{rows, cols}] -> vals, {N, N}], b}];
ddSolve[dd_, Vs_: 0, Vd_: 0, Vg_: 0, Vb_: 0, maxGummel_: 80, tol_: 1*^-5] := Module[
  {ps, Ny, Nx, phi, n, p, phiPrev, A, b, phiN, phiP, phiR, dphi, it},
  ps = dd["ps"]; Ny = ps["Ny"]; Nx = ps["Nx"];
  phiR = poissonSolve[ps, Vs, Vd, Vg, Vb];
  phi = phiR["phi"];
  n = Max[phiR["n"], 1.0]; p = Max[phiR["p"], 1.0];
  phiPrev = phi;
  Do[
    {A, b} = ddAssemble[dd, phi, n, p, "n"];
    n = Max[ArrayReshape[LinearSolve[A, b], {Ny, Nx}], 1*^-3];
    {A, b} = ddAssemble[dd, phi, n, p, "p"];
    p = Max[ArrayReshape[LinearSolve[A, b], {Ny, Nx}], 1*^-3];
    phiN = phi - Vt*Log[n/ni]; phiP = phi + Vt*Log[p/ni];
    phiR = poissonSolve[ps, Vs, Vd, Vg, Vb, phi, phiN, phiP];
    phi = phiR["phi"];
    dphi = Max[Abs[phi - phiPrev]]; phiPrev = phi;
    If[dphi < tol, Break[]],
    {it, 1, maxGummel}];
  <|"phi" -> phi, "n" -> n, "p" -> p, "iter" -> it, "dphi" -> dphi|>];
ddSolveHoles[dd_, phi_, nFixed_, pInit_, gen_, maxIter_: 30, tol_: 1*^-5] := Module[
  {Ny, Nx, p, A, b, pNew, err, it},
  Ny = dd["Ny"]; Nx = dd["Nx"];
  p = Max[pInit, 1*^-3];
  Do[
    {A, b} = ddAssemble[dd, phi, nFixed, p, "p", gen];
    pNew = Max[ArrayReshape[LinearSolve[A, b], {Ny, Nx}], 1*^-3];
    err = Max[Abs[Log[Max[pNew, 1*^-300]] - Log[Max[p, 1*^-300]]]];
    p = pNew;
    If[err < tol, Break[]],
    {it, 1, maxIter}];
  p];
ddDrainCurrent[dd_, phi_, n_] := Module[
  {ps, Nx, Ny, Ex, Ey, mux, muy, Id, faces, C, Dv, flux, drainNodes, dn, fc, jj, ii, mu, hh, Af, j, i},
  ps = dd["ps"]; Nx = ps["Nx"]; Ny = ps["Ny"];
  Ex = Table[Abs[phi[[j, i + 1]] - phi[[j, i]]]/ps["dxe"][[i]], {j, 1, Ny}, {i, 1, Nx - 1}];
  mux = mobField[0.5 (dd["munLF"][[All, 1 ;; -2]] + dd["munLF"][[All, 2 ;;]]), Ex, "n"];
  Ey = Table[Abs[phi[[j + 1, i]] - phi[[j, i]]]/ps["dyn"][[j]], {j, 1, Ny - 1}, {i, 1, Nx}];
  muy = mobField[0.5 (dd["munLF"][[1 ;; -2, All]] + dd["munLF"][[2 ;;, All]]), Ey, "n"];
  Id = 0.0;
  drainNodes = Position[ps["contactType"], 2];
  Do[
    {j, i} = dn;
    faces = {};
    If[i + 1 <= Nx, AppendTo[faces, {j, i + 1, mux[[j, i]], ps["dxe"][[i]], ps["dyc"][[j]]}]];
    If[i - 1 >= 1, AppendTo[faces, {j, i - 1, mux[[j, i - 1]], ps["dxe"][[i - 1]], ps["dyc"][[j]]}]];
    If[j + 1 <= Ny, AppendTo[faces, {j + 1, i, muy[[j, i]], ps["dyn"][[j]], ps["dxc"][[i]]}]];
    Do[
      {jj, ii, mu, hh, Af} = fc;
      C = q*mu*Vt*Af/hh;
      Dv = (phi[[jj, ii]] - phi[[j, i]])/Vt;
      flux = C*(bernV[Dv]*n[[jj, ii]] - bernV[-Dv]*n[[j, i]]);
      Id += flux,
      {fc, faces}],
    {dn, drainNodes}];
  Id*UM];

(* =====[SECTION: S6 bulk]===== *)
scCOP = 1/(1*^-13*dosFn[0.15]);
scEOP = Eop*eV;
scTAU1[epsEV_] := Module[{g, gp}, g = gammaFn[epsEV]; gp = gammaPFn[epsEV]; 1*^-14/(Sqrt[Max[g, 1*^-6]]*gp)];
scDcoef[epsEV_] := velFn[epsEV]^2*scTAU1[epsEV]/3;
V_Sm = 9.04*^3;                (* m/s *)
scAcoef[epsEV_] := 4 MSTAR*V_Sm^2*(epsEV*eV)/scTAU1[epsEV];
iiRateFn[epsEV_] := Module[{x}, x = Max[(epsEV - 1.10)/1.10, 0.0]; 2.0*^13*x^2];
solveBulk[EVcm_, epsMaxEV_: 3.02, ne_: 1209, absorbingTop_: True] := Module[
  {eps, depsEV, depsJ, Z, D, ESI, qE, Deps, ZDeps, face, delta, Bp, Bm,
   rows, cols, vals, diagV, Ldiff, m, xop, Nop, qrows, qcols, qvals, qdiag,
   Qmat, ZA, ZAface, arows, acols, avals, adiag, Lac, M, b, f0, nInt, i, c},
  eps = Range[1*^-4, epsMaxEV, (epsMaxEV - 1*^-4)/(ne - 1)];
  depsEV = eps[[2]] - eps[[1]]; depsJ = depsEV*eV;
  Z = Table[dosFn[e], {e, eps}];
  D = Table[scDcoef[e], {e, eps}];
  ESI = EVcm*100.0; qE = q*ESI;
  Deps = qE^2*D;
  ZDeps = Z*Deps;
  face = 0.5 (ZDeps[[1 ;; -2]] + ZDeps[[2 ;;]]);
  rows = {}; cols = {}; vals = {}; diagV = ConstantArray[0.0, ne];
  Do[
    If[i + 1 <= ne, c = face[[i]]/depsJ;
      AppendTo[rows, i]; AppendTo[cols, i + 1]; AppendTo[vals, -c]; diagV[[i]] += c];
    If[i - 1 >= 1, c = face[[i - 1]]/depsJ;
      AppendTo[rows, i]; AppendTo[cols, i - 1]; AppendTo[vals, -c]; diagV[[i]] += c],
    {i, 1, ne}];
  Ldiff = SparseArray[Transpose[{rows, cols}] -> vals, {ne, ne}] + SparseArray[Range[ne] -> diagV, {ne}];
  m = Round[scEOP/depsJ];
  xop = scEOP/(kB T); Nop = 1/ExpM1[xop];
  qrows = {}; qcols = {}; qvals = {}; qdiag = ConstantArray[0.0, ne];
  Do[
    ip = i + m; im = i - m;
    If[ip <= ne,
      AppendTo[qrows, i]; AppendTo[qcols, ip]; AppendTo[qvals, scCOP*Z[[i]]*(Nop + 1)*Z[[ip]]];
      qdiag[[i]] += -scCOP*Z[[i]]*Nop*Z[[ip]]];
    If[im >= 1,
      AppendTo[qrows, i]; AppendTo[qcols, im]; AppendTo[qvals, scCOP*Z[[i]]*Nop*Z[[im]]];
      qdiag[[i]] += -scCOP*Z[[i]]*(Nop + 1)*Z[[im]]],
    {i, 1, ne}];
  Qmat = SparseArray[Transpose[{qrows, qcols}] -> qvals, {ne, ne}] + SparseArray[Range[ne] -> qdiag, {ne}];
  ZA = Z*Table[scAcoef[e], {e, eps}];
  ZAface = 0.5 (ZA[[1 ;; -2]] + ZA[[2 ;;]]);
  delta = depsJ/(kB T); Bp = bernV[delta]; Bm = bernV[-delta];
  arows = {}; acols = {}; avals = {}; adiag = ConstantArray[0.0, ne];
  Do[
    If[i + 1 <= ne, c = ZAface[[i]]/depsJ;
      AppendTo[arows, i]; AppendTo[acols, i + 1]; AppendTo[avals, -c*Bm]; adiag[[i]] += c*Bp];
    If[i - 1 >= 1, c = ZAface[[i - 1]]/depsJ;
      AppendTo[arows, i]; AppendTo[acols, i - 1]; AppendTo[avals, -c*Bp]; adiag[[i]] += c*Bm],
    {i, 1, ne}];
  Lac = SparseArray[Transpose[{arows, acols}] -> avals, {ne, ne}] + SparseArray[Range[ne] -> adiag, {ne}];
  M = Ldiff + Lac - depsJ*Qmat;
  M = ReplacePart[M, {2, 2} -> 1.0];
  b = ConstantArray[0.0, ne]; b[[2]] = 1.0;
  If[absorbingTop, M = ReplacePart[M, {ne, ne} -> 1.0]; b[[ne]] = 0.0];
  f0 = LinearSolve[M, b];
  f0 = Max[f0, 1*^-300];
  nInt = trap1D[Z*f0, eps*eV];
  {eps, f0/nInt}];


Print["SMOKE S1-S6 OK"];

