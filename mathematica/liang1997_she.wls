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
Off[RuleDelayed::rhs]; Off[General::stop]; Off[Part::partw];
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

(* =====[SECTION: S7 she2d]===== *)
trapzLast[a3_, h_] := Module[{w}, w = Differences[h];
  Total[0.5 (a3[[;;, ;;, 2 ;;]] + a3[[;;, ;;, ;; -2]])*ArrayReshape[w, {1, 1, Length[w]}], {3}]];
pad3[w_, nh_] := Transpose[ConstantArray[w, {nh}], {2, 3, 1}];
hgrid[phi_, dH_, mpad_] := Module[{phiMin, phiMax, Hlo, Hhi, count},
  phiMin = Min[phi]; phiMax = Max[phi];
  Hlo = (-phiMax)*eV - mpad*dH;
  Hhi = EPSMAX + (-phiMin)*eV + mpad*dH;
  count = Ceiling[(Hhi + 0.5 dH - Hlo)/dH];
  Hlo + (Range[count] - 1)*dH];
sheInit[xum_, yum_, phi_, ctype_, neq_, dHeV_: 0.0125, withII_: True, absTop_: True, phiSpan_: None] := Module[
  {xm, ym, Nx, Ny, dH, mop, phiMin, phiMax, mpad, H, NH, dxc, dyc, dxe, dyn, area,
   eps3, active, epsE, Z3, D3, Aac3, ii3, gidFlat, pos, gid3, nDOF, ZDtop, ZAtop, ctypeI, interior3},
  xm = xum*1*^-6; ym = yum*1*^-6;
  Nx = Length[xm]; Ny = Length[ym];
  dH = dHeV*eV;
  mop = Round[Eop*eV/dH];
  If[Abs[mop*dH - Eop*eV] > 1*^-12*eV, Print["E_OP/dH not integer!"]; Abort[]];
  {phiMin, phiMax} = If[phiSpan === None, {Min[phi], Max[phi]}, phiSpan];
  mpad = Max[2, mop];
  H = hgrid[phi, dH, mpad];
  NH = Length[H];
  dxc = Table[If[i == 1, 0.5 (xm[[2]] - xm[[1]]), If[i == Nx, 0.5 (xm[[Nx]] - xm[[Nx - 1]]), 0.5 (xm[[i + 1]] - xm[[i - 1]])]], {i, 1, Nx}];
  dyc = Table[If[j == 1, 0.5 (ym[[2]] - ym[[1]]), If[j == Ny, 0.5 (ym[[Ny]] - ym[[Ny - 1]]), 0.5 (ym[[j + 1]] - ym[[j - 1]])]], {j, 1, Ny}];
  dxe = Differences[xm]; dyn = Differences[ym];
  area = Outer[Times, dyc, dxc];
  eps3 = ArrayReshape[Outer[Plus, Flatten[phi*eV], H], {Ny, Nx, NH}];
  active = UnitStep[eps3]*(1 - UnitStep[eps3 - EPSMAX]);
  epsE = Map[Clip[#, {1*^-8, EPSMAX/eV}] &, eps3/eV, {3}];
  Z3 = active*Map[dosFn, epsE, {3}];
  D3 = active*Map[scDcoef, epsE, {3}];
  Aac3 = active*Map[scAcoef, epsE, {3}];
  ii3 = active*Map[iiRateFn, epsE, {3}];
  gidFlat = ConstantArray[-1, Ny*Nx*NH];
  pos = Flatten[Position[Flatten[active], 1]];
  gidFlat[[pos]] = Range[Length[pos]];
  nDOF = Length[pos];
  gid3 = ArrayReshape[gidFlat, {Ny, Nx, NH}];
  ZDtop = dosFn[EPSMAX/eV]*scDcoef[EPSMAX/eV];
  ZAtop = dosFn[EPSMAX/eV]*scAcoef[EPSMAX/eV];
  ctypeI = Boole[Map[# == 0 &, ctype, {2}]];
  interior3 = pad3[ctypeI, NH];
  <|"x" -> xum, "y" -> yum, "phi" -> phi, "ctype" -> ctype, "neq" -> neq,
    "Nx" -> Nx, "Ny" -> Ny, "NH" -> NH, "dH" -> dH, "mop" -> mop, "H" -> H,
    "eps3" -> eps3, "active" -> active, "Z" -> Z3, "D" -> D3, "Aac" -> Aac3,
    "ii" -> ii3, "gid3" -> gid3, "gidFlat" -> gidFlat, "nDOF" -> nDOF,
    "dxc" -> dxc, "dyc" -> dyc, "dxe" -> dxe, "dyn" -> dyn, "area" -> area,
    "ZDtop" -> ZDtop, "ZAtop" -> ZAtop, "interior3" -> interior3,
    "includeII" -> withII, "absorbingTop" -> absTop|>];

sheSolve[she_] := Module[
  {Ny, Nx, NH, m, dH, Z3, D3, Aac3, ii3, eps3, gid3, gidFlat, nDOF, active, interior3,
   H, Aij3, Nop, cop, delta, Bp, Bm, ZD3, offParts, diagParts, b, cutSp, cutAc, cutOp,
   iiCoeff, ctL, A, Dv, As, bs, sol, fRaw, f, F3f, F3, n, num, den, good, Te, Gii,
   comX, Kx, dfdx, gxH, gx, comY, Ky, dfdy, gyH, gy, Gx, Gy, vx, vy, rates,
   iiEv, cutTot, contactIn, balance, scale, t0},
  Ny = she["Ny"]; Nx = she["Nx"]; NH = she["NH"]; m = she["mop"]; dH = she["dH"];
  Z3 = she["Z"]; D3 = she["D"]; Aac3 = she["Aac"]; ii3 = she["ii"]; eps3 = she["eps3"];
  gid3 = she["gid3"]; gidFlat = she["gidFlat"]; nDOF = she["nDOF"];
  active = she["active"]; interior3 = she["interior3"]; H = she["H"];
  Aij3 = pad3[she["area"], NH];
  Nop = 1/ExpM1[Eop*eV/(kB T)]; cop = scCOP;
  delta = dH/(kB T); Bp = bernV[delta]; Bm = bernV[-delta];
  ZD3 = Z3*D3;
  offParts = {}; diagParts = {}; cutSp = {}; cutAc = {}; cutOp = {}; iiCoeff = {};
  ctL = {};
  b = ConstantArray[0.0, nDOF];
  (* ---------- spatial diffusion edges ---------- *)
  addEdge[dir_] := Module[{L3, R3, actL, actR, ZDL, ZDR, Tf3, ww, gL, gR, cL3, cR3,
     maskII, maskIC, maskCI, fL, r1, c1, v1, fIC, rIC, cIC, vIC, ctIC,
     fCI, rCI, cCI, vCI, ctCI, nnn},
    Which[
      dir == "E", L3 = interior3[[All, 1 ;; -2, All]]; R3 = interior3[[All, 2 ;;, All]];
        actL = active[[All, 1 ;; -2, All]]; actR = active[[All, 2 ;;, All]];
        ZDL = ZD3[[All, 1 ;; -2, All]]; ZDR = ZD3[[All, 2 ;;, All]];
        ww = Outer[Divide, she["dyc"], she["dxe"]];
        gL = gid3[[All, 1 ;; -2, All]]; gR = gid3[[All, 2 ;;, All]];
        cL3 = pad3[she["ctype"][[All, 1 ;; -2]], NH]; cR3 = pad3[she["ctype"][[All, 2 ;;]], NH],
      dir == "W", L3 = interior3[[All, 2 ;;, All]]; R3 = interior3[[All, 1 ;; -2, All]];
        actL = active[[All, 2 ;;, All]]; actR = active[[All, 1 ;; -2, All]];
        ZDL = ZD3[[All, 2 ;;, All]]; ZDR = ZD3[[All, 1 ;; -2, All]];
        ww = Outer[Divide, she["dyc"], she["dxe"]];
        gL = gid3[[All, 2 ;;, All]]; gR = gid3[[All, 1 ;; -2, All]];
        cL3 = pad3[she["ctype"][[All, 2 ;;]], NH]; cR3 = pad3[she["ctype"][[All, 1 ;; -2]], NH],
      dir == "N", L3 = interior3[[1 ;; -2, All, All]]; R3 = interior3[[2 ;;, All, All]];
        actL = active[[1 ;; -2, All, All]]; actR = active[[2 ;;, All, All]];
        ZDL = ZD3[[1 ;; -2, All, All]]; ZDR = ZD3[[2 ;;, All, All]];
        ww = Outer[Times, 1/she["dyn"], she["dxc"]];
        gL = gid3[[1 ;; -2, All, All]]; gR = gid3[[2 ;;, All, All]];
        cL3 = pad3[she["ctype"][[1 ;; -2, All]], NH]; cR3 = pad3[she["ctype"][[2 ;;, All]], NH],
      dir == "S", L3 = interior3[[2 ;;, All, All]]; R3 = interior3[[1 ;; -2, All, All]];
        actL = active[[2 ;;, All, All]]; actR = active[[1 ;; -2, All, All]];
        ZDL = ZD3[[2 ;;, All, All]]; ZDR = ZD3[[1 ;; -2, All, All]];
        ww = Outer[Times, 1/she["dyn"], she["dxc"]];
        gL = gid3[[2 ;;, All, All]]; gR = gid3[[1 ;; -2, All, All]];
        cL3 = pad3[she["ctype"][[2 ;;, All]], NH]; cR3 = pad3[she["ctype"][[1 ;; -2, All]], NH]];
    Tf3 = dH*0.5 (ZDL + ZDR)*pad3[ww, NH];
    maskII = actL*actR*L3*R3;
    maskIC = actL*actR*L3*(1 - R3);
    maskCI = actL*actR*(1 - L3)*R3;
    fL = Flatten[maskII];
    r1 = Pick[Flatten[gL], fL, 1]; c1 = Pick[Flatten[gR], fL, 1]; v1 = Pick[Flatten[Tf3], fL, 1];
    If[Length[r1] > 0,
      offParts = Append[offParts, SparseArray[Transpose[{r1, c1}] -> -v1, {nDOF, nDOF}]];
      offParts = Append[offParts, SparseArray[Transpose[{c1, r1}] -> -v1, {nDOF, nDOF}]];
      diagParts = Append[diagParts, SparseArray[Transpose[{r1}] -> v1, {nDOF}]];
      diagParts = Append[diagParts, SparseArray[Transpose[{c1}] -> v1, {nDOF}]]];
    fIC = Flatten[maskIC];
    rIC = Pick[Flatten[gL], fIC, 1]; cIC = Pick[Flatten[gR], fIC, 1]; vIC = Pick[Flatten[Tf3], fIC, 1];
    If[Length[rIC] > 0,
      offParts = Append[offParts, SparseArray[Transpose[{rIC, cIC}] -> -vIC, {nDOF, nDOF}]];
      diagParts = Append[diagParts, SparseArray[Transpose[{rIC}] -> vIC, {nDOF}]];
      ctIC = Pick[Flatten[cR3], fIC, 1];
      ctL = Join[ctL, Transpose[{ctIC, rIC, cIC, vIC}]]];
    fCI = Flatten[maskCI];
    rCI = Pick[Flatten[gR], fCI, 1]; cCI = Pick[Flatten[gL], fCI, 1]; vCI = Pick[Flatten[Tf3], fCI, 1];
    If[Length[rCI] > 0,
      offParts = Append[offParts, SparseArray[Transpose[{rCI, cCI}] -> -vCI, {nDOF, nDOF}]];
      diagParts = Append[diagParts, SparseArray[Transpose[{rCI}] -> vCI, {nDOF}]];
      ctCI = Pick[Flatten[cL3], fCI, 1];
      ctL = Join[ctL, Transpose[{ctCI, rCI, cCI, vCI}]]];
    ];
  addAbsEdge[dir_] := Module[{L3, actL, actR, epsL, epsR, ZDL, ww, gL, hc3, maskA, de, frac,
     dist, Kface, Tf, fA, rA, vA},
    Which[
      dir == "E", L3 = interior3[[All, 1 ;; -2, All]]; actL = active[[All, 1 ;; -2, All]];
        actR = active[[All, 2 ;;, All]];
        epsL = eps3[[All, 1 ;; -2, All]]; epsR = eps3[[All, 2 ;;, All]];
        ZDL = ZD3[[All, 1 ;; -2, All]]; ww = Outer[Divide, she["dyc"], she["dxe"]];
        gL = gid3[[All, 1 ;; -2, All]]; hc3 = pad3[ConstantArray[she["dxe"], Ny], NH],
      dir == "W", L3 = interior3[[All, 2 ;;, All]]; actL = active[[All, 2 ;;, All]];
        actR = active[[All, 1 ;; -2, All]];
        epsL = eps3[[All, 2 ;;, All]]; epsR = eps3[[All, 1 ;; -2, All]];
        ZDL = ZD3[[All, 2 ;;, All]]; ww = Outer[Divide, she["dyc"], she["dxe"]];
        gL = gid3[[All, 2 ;;, All]]; hc3 = pad3[ConstantArray[she["dxe"], Ny], NH],
      dir == "N", L3 = interior3[[1 ;; -2, All, All]]; actL = active[[1 ;; -2, All, All]];
        actR = active[[2 ;;, All, All]];
        epsL = eps3[[1 ;; -2, All, All]]; epsR = eps3[[2 ;;, All, All]];
        ZDL = ZD3[[1 ;; -2, All, All]]; ww = Outer[Times, 1/she["dyn"], she["dxc"]];
        gL = gid3[[1 ;; -2, All, All]]; hc3 = pad3[Transpose[ConstantArray[she["dyn"], Nx]], NH],
      dir == "S", L3 = interior3[[2 ;;, All, All]]; actL = active[[2 ;;, All, All]];
        actR = active[[1 ;; -2, All, All]];
        epsL = eps3[[2 ;;, All, All]]; epsR = eps3[[1 ;; -2, All, All]];
        ZDL = ZD3[[2 ;;, All, All]]; ww = Outer[Times, 1/she["dyn"], she["dxc"]];
        gL = gid3[[2 ;;, All, All]]; hc3 = pad3[Transpose[ConstantArray[she["dyn"], Nx]], NH]];
    maskA = actL*(1 - actR)*UnitStep[epsR - EPSMAX];
    de = epsR - epsL;
    frac = Map[Clip[#, {1*^-8, 1.0}] &, (EPSMAX - epsL)/de, {3}];
    dist = frac*hc3;
    Kface = 0.5 (ZDL + she["ZDtop"]);
    Tf = dH*Kface*pad3[ww, NH]/dist;
    fA = Flatten[maskA]; rA = Pick[Flatten[gL], fA, 1]; vA = Pick[Flatten[Tf], fA, 1];
    If[Length[rA] > 0,
      diagParts = Append[diagParts, SparseArray[Transpose[{rA}] -> vA, {nDOF}]];
      cutSp = Join[cutSp, Transpose[{rA, vA}]]];
    ];
  addEdge["E"]; addEdge["W"]; addEdge["N"]; addEdge["S"];
  addAbsEdge["E"]; addAbsEdge["W"]; addAbsEdge["N"]; addAbsEdge["S"];
  (* ---------- optical phonons ---------- *)
  (* emission: k -> k-m *)
  curD = gid3[[All, All, 1 + m ;; All]]; tgtD = gid3[[All, All, 1 ;; NH - m]];
  actD = active[[All, All, 1 + m ;; All]]*active[[All, All, 1 ;; NH - m]]*interior3[[All, All, 1 + m ;; All]];
  ZkD = Z3[[All, All, 1 + m ;; All]]; Zd = Z3[[All, All, 1 ;; NH - m]];
  wD = Aij3[[All, All, 1 + m ;; All]]*dH*cop*ZkD*(Nop + 1)*Zd;
  uD = Aij3[[All, All, 1 + m ;; All]]*dH*cop*ZkD*Nop*Zd;
  fD = Flatten[actD]; rD = Pick[Flatten[curD], fD, 1]; cD = Pick[Flatten[tgtD], fD, 1];
  vD = Pick[Flatten[wD], fD, 1]; uDf = Pick[Flatten[uD], fD, 1];
  If[Length[rD] > 0,
    offParts = Append[offParts, SparseArray[Transpose[{rD, cD}] -> -uDf, {nDOF, nDOF}]];
    diagParts = Append[diagParts, SparseArray[Transpose[{rD}] -> vD, {nDOF}]]];
  (* absorption: k -> k+m *)
  curU = gid3[[All, All, 1 ;; NH - m]]; tgtU = gid3[[All, All, 1 + m ;; All]];
  actU = active[[All, All, 1 ;; NH - m]]*active[[All, All, 1 + m ;; All]]*interior3[[All, All, 1 ;; NH - m]];
  ZkU = Z3[[All, All, 1 ;; NH - m]]; Zu = Z3[[All, All, 1 + m ;; All]];
  wU = Aij3[[All, All, 1 ;; NH - m]]*dH*cop*ZkU*Nop*Zu;
  uU = Aij3[[All, All, 1 ;; NH - m]]*dH*cop*ZkU*(Nop + 1)*Zu;
  fU = Flatten[actU]; rU = Pick[Flatten[curU], fU, 1]; cU = Pick[Flatten[tgtU], fU, 1];
  vU = Pick[Flatten[wU], fU, 1]; uUf = Pick[Flatten[uU], fU, 1];
  If[Length[rU] > 0,
    offParts = Append[offParts, SparseArray[Transpose[{rU, cU}] -> -uUf, {nDOF, nDOF}]];
    diagParts = Append[diagParts, SparseArray[Transpose[{rU}] -> vU, {nDOF}]]];
  (* absorption beyond eps_max -> numerical reservoir *)
  actOA = active*UnitStep[eps3 + Eop*eV - EPSMAX]*interior3;
  ZuA = Map[dosFn, (eps3 + Eop*eV)/eV, {3}];
  vOA = Aij3*dH*cop*Z3*Nop*ZuA;
  fOA = Flatten[actOA]; rOA = Pick[Flatten[gid3], fOA, 1]; vOAf = Pick[Flatten[vOA], fOA, 1];
  If[Length[rOA] > 0,
    diagParts = Append[diagParts, SparseArray[Transpose[{rOA}] -> vOAf, {nDOF}]];
    cutOp = Join[cutOp, Transpose[{rOA, vOAf}]]];
  (* ---------- acoustic Fokker-Planck (conservative SG) ---------- *)
  ZAc = Z3*Aac3;
  actAc = active[[All, All, 1 ;; -1]]*active[[All, All, 2 ;;]]*interior3[[All, All, 1 ;; -1]];
  ZAf = 0.5 (ZAc[[All, All, 1 ;; -1]] + ZAc[[All, All, 2 ;;]])/dH;
  gAcL = gid3[[All, All, 1 ;; -1]]; gAcR = gid3[[All, All, 2 ;;]];
  vAcD = Aij3[[All, All, 1 ;; -1]]*ZAf*Bp;
  vAcO = Aij3[[All, All, 1 ;; -1]]*ZAf*Bm;
  fAc = Flatten[actAc]; rAcL = Pick[Flatten[gAcL], fAc, 1]; rAcR = Pick[Flatten[gAcR], fAc, 1];
  vAcDf = Pick[Flatten[vAcD], fAc, 1]; vAcOf = Pick[Flatten[vAcO], fAc, 1];
  If[Length[rAcL] > 0,
    offParts = Append[offParts, SparseArray[Transpose[{rAcL, rAcR}] -> -vAcOf, {nDOF, nDOF}]];
    offParts = Append[offParts, SparseArray[Transpose[{rAcR, rAcL}] -> -vAcDf, {nDOF, nDOF}]];
    diagParts = Append[diagParts, SparseArray[Transpose[{rAcL}] -> vAcDf, {nDOF}]];
    diagParts = Append[diagParts, SparseArray[Transpose[{rAcR}] -> vAcOf, {nDOF}]]];
  (* acoustic absorbing up *)
  actN = Join[active[[All, All, 2 ;;]], 0 active[[All, All, -1 ;;]], 3];
  epsN = Join[eps3[[All, All, 2 ;;]], eps3[[All, All, -1 ;;]] + dH, 3];
  maskAA = interior3*active*(1 - actN)*UnitStep[epsN - EPSMAX];
  deB = EPSMAX - eps3;
  ZAface2 = 0.5 (ZAc + she["ZAtop"]);
  cb = ZAface2/Map[Max[#, 1*^-12 dH] &, deB, {3}];
  Bpb = bernV[deB/(kB T)];
  ccut = Aij3*cb*Bpb;
  fAA = Flatten[maskAA]; rAA = Pick[Flatten[gid3], fAA, 1]; vAA = Pick[Flatten[ccut], fAA, 1];
  If[Length[rAA] > 0,
    diagParts = Append[diagParts, SparseArray[Transpose[{rAA}] -> vAA, {nDOF}]];
    cutAc = Join[cutAc, Transpose[{rAA, vAA}]]];
  (* ---------- impact ionization ---------- *)
  If[she["includeII"],
    actII = interior3*active*UnitStep[ii3 - 1*^-300];
    fII = Flatten[actII]; posII = Flatten[Position[fII, 1]];
    If[Length[posII] > 0,
      epsS = Pick[Flatten[eps3], fII, 1];
      epsTgt = 0.5 (epsS - 1.10*eV);
      cellF = Ceiling[posII/NH];
      phiF = Pick[Flatten[pad3[she["phi"], NH]], fII, 1];
      Ht = epsTgt - phiF*eV;
      uu = (Ht - H[[1]])/dH;
      k0m = Floor[uu]; a = uu - k0m;
      kA = k0m + 1; kB = k0m + 2;
      actF = Flatten[active];
      pA = (cellF - 1)*NH + Clip[kA, {1, NH}];
      pB = (cellF - 1)*NH + Clip[kB, {1, NH}];
      vA = (1 - a); vB = a*UnitStep[a - 1*^-16];
      okA = UnitStep[kA - 1]*UnitStep[NH - kA]*actF[[pA]];
      okB = UnitStep[kB - 1]*UnitStep[NH - kB]*actF[[pB]];
      wA = vA*okA; wB = vB*okB;
      sw = wA + wB;
      keep = UnitStep[sw - 1*^-300];
      cii = Pick[Flatten[Aij3*dH*Z3*ii3], fII, 1]*keep;
      If[Total[keep] > 0,
        diagParts = Append[diagParts, SparseArray[Transpose[{posII}] -> cii, {nDOF}]];
        iiCoeff = Join[iiCoeff, Transpose[{posII, cii}]];
        sws = Max[sw, 1*^-300];
        wAn = wA/sws; wBn = wB/sws;
        gA = gidFlat[[pA]]; gB = gidFlat[[pB]];
        mA = UnitStep[gA + 0.5]*UnitStep[cii - 1*^-300];
        mB = UnitStep[gB + 0.5]*UnitStep[cii - 1*^-300];
        offParts = Append[offParts, SparseArray[Transpose[{Pick[gA, mA, 1], Pick[posII, mA, 1]}] -> Pick[-2 wAn*cii, mA, 1], {nDOF, nDOF}]];
        offParts = Append[offParts, SparseArray[Transpose[{Pick[gB, mB, 1], Pick[posII, mB, 1]}] -> Pick[-2 wBn*cii, mB, 1], {nDOF, nDOF}]]];
      ]];
  (* ---------- contacts: Maxwellian Dirichlet rows ---------- *)
  ct3 = pad3[Boole[Map[# >= 1 &, she["ctype"], {2}]], NH];
  neq3 = pad3[she["neq"]*1*^6, NH];
  contMask = ct3*active;
  contPos = Flatten[Position[Flatten[contMask], 1]];
  If[Length[contPos] > 0,
    epsC = Pick[Flatten[eps3], Flatten[contMask], 1];
    ZC = Pick[Flatten[Z3], Flatten[contMask], 1];
    neqC = Pick[Flatten[neq3], Flatten[contMask], 1];
    gC = Pick[Flatten[gid3], Flatten[contMask], 1];
    cellC = Ceiling[contPos/NH];
    ZcV = Table[0.0, {Length[cellC]}];
    Do[
      jj = Ceiling[cellC[[cc]]/Nx]; ii = Mod[cellC[[cc]] - 1, Nx] + 1;
      ZcV[[cc]] = trap1D[Table[Z3[[jj, ii, kk]]*Exp[-Min[eps3[[jj, ii, kk]]/(kB T), 200]], {kk, 1, NH}], H],
      {cc, 1, Length[cellC]}];
    bC = neqC*Exp[-Map[Min[#, 200] &, epsC/(kB T), {1}]]/(ZcV + 1*^-300);
    b[[gC]] = bC;
    diagParts = Append[diagParts, SparseArray[Transpose[{gC}] -> 1.0, {nDOF}]];
    ];
  (* ---------- assemble + solve ---------- *)
  A = Total[offParts] + SparseArray[Range[nDOF] -> Total[diagParts], {nDOF, nDOF}];
  Dv = A["Diagonal"];
  Dv[[Flatten[Position[Dv, x_ /; Abs[x] < 1*^-300]]]] = 1.0;
  As = DiagonalMatrix[SparseArray[1/Dv]].A;
  bs = b/Dv;
  Print["S7 SHE: DOF=", nDOF, "  NH=", NH, "  nnz=", Length[A["NonzeroValues"]]];
  t0 = AbsoluteTime[];
  sol = LinearSolve[As, bs];
  Print["S7 SHE: direct sparse solve ", Round[AbsoluteTime[] - t0], " s"];
  fRaw = sol;
  f = Max[fRaw, 0.0];
  F3f = ConstantArray[0.0, Ny*Nx*NH];
  F3f[[Flatten[Position[Flatten[active], 1]]]] = f;
  F3 = ArrayReshape[F3f, {Ny, Nx, NH}];
  (* ---------- moments ---------- *)
  n = trapzLast[Z3*F3, H]*1*^-6;
  num = trapzLast[eps3*Z3*F3, H];
  den = trapzLast[Z3*F3, H];
  good = UnitStep[n*1*^6 - 1*^19];
  Te = (2/3)*num/(den + 1*^-300)/kB*good + T*(1 - good);
  Gii = trapzLast[Z3*F3*ii3, H]*1*^-6;
  comX = active[[All, 1 ;; -2, All]]*active[[All, 2 ;;, All]];
  Kx = 0.5 (ZD3[[All, 1 ;; -2, All]] + ZD3[[All, 2 ;;, All]]);
  dfdx = (F3[[All, 2 ;;, All]] - F3[[All, 1 ;; -2, All]])*ArrayReshape[1/she["dxe"], {1, Nx - 1, 1}];
  gxH = -Kx*dfdx*comX;
  gx = trapzLast[gxH, H];
  comY = active[[1 ;; -2, All, All]]*active[[2 ;;, All, All]];
  Ky = 0.5 (ZD3[[1 ;; -2, All, All]] + ZD3[[2 ;;, All, All]]);
  dfdy = (F3[[2 ;;, All, All]] - F3[[1 ;; -2, All, All]])*ArrayReshape[1/she["dyn"], {Ny - 1, 1, 1}];
  gyH = -Ky*dfdy*comY;
  gy = trapzLast[gyH, H];
  Gx = ConstantArray[0.0, {Ny, Nx}]; Gy = ConstantArray[0.0, {Ny, Nx}];
  If[Nx > 1, Gx[[All, 1]] = gx[[All, 1]]; Gx[[All, -1]] = gx[[All, -1]]];
  If[Nx > 2, Gx[[All, 2 ;; -2]] = 0.5 (gx[[All, 1 ;; -2]] + gx[[All, 2 ;; -1]])];
  If[Ny > 1, Gy[[1, All]] = gy[[1, All]]; Gy[[-1, All]] = gy[[-1, All]]];
  If[Ny > 2, Gy[[2 ;; -2, All]] = 0.5 (gy[[1 ;; -2, All]] + gy[[2 ;; -1, All]])];
  vx = Gx/Map[Max[#, 1*^-300] &, n*1*^6, {2}]*100*good;
  vy = Gy/Map[Max[#, 1*^-300] &, n*1*^6, {2}]*100*good;
  (* ---------- terminal rates / conservation ---------- *)
  rates = <|1 -> 0.0, 2 -> 0.0, 3 -> 0.0|>;
  If[Length[ctL] > 0,
    Do[rates[[ctL[[kk, 1]]]] += ctL[[kk, 4]]*(f[[ctL[[kk, 3]]]] - f[[ctL[[kk, 2]]]]), {kk, 1, Length[ctL]}]];
  iiEv = If[Length[iiCoeff] > 0, Total[iiCoeff[[All, 2]]*f[[iiCoeff[[All, 1]]]]], 0.0];
  cutTot = 0.0;
  If[Length[cutSp] > 0, cutTot += Total[cutSp[[All, 2]]*f[[cutSp[[All, 1]]]]]];
  If[Length[cutAc] > 0, cutTot += Total[cutAc[[All, 2]]*f[[cutAc[[All, 1]]]]]];
  If[Length[cutOp] > 0, cutTot += Total[cutOp[[All, 2]]*f[[cutOp[[All, 1]]]]]];
  contactIn = Total[Values[rates]];
  balance = contactIn + iiEv - cutTot;
  scale = Max[Abs[contactIn], Abs[iiEv], Abs[cutTot], 1*^-300];
  <|"f" -> f, "fRaw" -> fRaw, "F3" -> F3, "n" -> n, "Te" -> Te, "Gii" -> Gii,
    "vx" -> vx, "vy" -> vy, "gx" -> gx, "gy" -> gy,
    "rates" -> rates,
    "currents" -> <|1 -> -q*rates[1]*1*^-6, 2 -> -q*rates[2]*1*^-6, 3 -> -q*rates[3]*1*^-6|>,
    "iiEvents" -> iiEv, "cutoff" -> cutTot, "contactIn" -> contactIn,
    "balance" -> balance, "relBalance" -> balance/scale, "contactLinks" -> ctL|>];

(* =====[SECTION: S8 coupled]===== *)
solveCoupled[Nx_, Ny_, Vg_: 3.0, Vd_: 3.0, maxOuter_: 3, phiDamp_: 0.25, PhiGate_: 0.30] := Module[
  {x, y, X, Y, Nd, Na, Nnet, ps, ddS, phi, res, pH, hist, it, she, rr, nSH, GiiSH,
   phiN, phiP, phiT, err},
  {x, y, X, Y} = makeMesh[Nx, Ny];
  {Nd, Na, Nnet} = buildDoping[{X, Y}];
  ps = poissonInit[x, y, Nnet, PhiGate];
  ddS = ddInit[ps];
  res = ddSolve[ddS, 0, Vd, Vg, 0, 90, 1*^-5];
  phi = res["phi"]; pH = res["p"];
  hist = {};
  Do[
    she = sheInit[x, y, phi, ps["contactType"], ddS["neq"]];
    rr = sheSolve[she];
    nSH = rr["n"]; GiiSH = rr["Gii"];
    pH = ddSolveHoles[ddS, phi, nSH, pH, GiiSH];
    phiN = phi - Vt*Log[Max[nSH, 1*^-30]/ni];
    phiP = phi + Vt*Log[Max[pH, 1*^-30]/ni];
    phiT = poissonSolve[ps, 0, Vd, Vg, 0, phi, phiN, phiP]["phi"];
    err = Max[Abs[phiDamp*(phiT - phi)]];
    phi = phi + phiDamp*(phiT - phi);
    AppendTo[hist, <|"it" -> it, "dphi" -> err, "nmax" -> Max[nSH],
      "Temax" -> Max[rr["Te"]], "Giimax" -> Max[GiiSH]|>];
    Print["S8 coupled outer ", it, ":  dphi=", ScientificForm[err, 2], "  nmax=",
      ScientificForm[Max[nSH], 3], "  Temax=", NumberForm[Max[rr["Te"]], 5],
      "  Giimax=", ScientificForm[Max[GiiSH], 3]],
    {it, 1, maxOuter}];
  hist];

(* =====[SECTION: S9 verification]===== *)
refDir = FileNameJoin[{dir, "ref"}];
figDir = FileNameJoin[{dir, "figures"}];
If[! DirectoryQ[figDir], CreateDirectory[figDir]];
importRef[name_] := Import[FileNameJoin[{refDir, name}], "CSV"];
xRef = Flatten[importRef["x.csv"]];
yRef = Flatten[importRef["y.csv"]];
HRef = Flatten[importRef["H_ref.csv"]];
phiRef = importRef["phi_ref.csv"];
nRef = importRef["n_ref.csv"];
TeRef = importRef["Te_ref.csv"];
GiiRef = importRef["Gii_ref.csv"];
vxRef = importRef["vx_ref.csv"];
vyRef = importRef["vy_ref.csv"];
ctypeRef = Round[importRef["ctype_ref.csv"]];
neqRef = importRef["neq_ref.csv"];

(* ---------- fig 1: bands ---------- *)
epsV = Range[1*^-4, 3.02, 3.02/399.0];
g1 = ListLinePlot[Table[{e, gammaFn[e]}, {e, epsV}], Frame -> True,
  FrameLabel -> {"eps [eV]", "gamma [eV]"}, PlotLabel -> "band function", ImageSize -> 260];
g2 = ListLinePlot[Table[{e, velFn[e]/1*^5}, {e, epsV}], Frame -> True,
  FrameLabel -> {"eps [eV]", "v [1e5 m/s]"}, PlotLabel -> "group velocity", ImageSize -> 260];
g3 = ListLogLinearPlot[Table[{e, dosPerEV[e]*1*^-6}, {e, epsV}], Frame -> True,
  FrameLabel -> {"eps [eV]", "Z [cm^-3 eV^-1]"}, PlotLabel -> "generalized DOS", ImageSize -> 260];
Export[FileNameJoin[{figDir, "fig1_bands.png"}], GraphicsGrid[{{g1, g2, g3}}], ImageResolution -> 110];

(* ---------- fig 2: device doping (160x140) ---------- *)
{x2, y2, X2, Y2} = makeMesh[160, 140];
{Nd2, Na2, Nnet2} = buildDoping[{X2, Y2}];
nAbs = Abs[Nnet2];
p1 = ArrayPlot[Reverse[Nnet2], ColorFunction -> (ColorData["RedBlueTones"][0.5 (1 + Sign[#] Min[Abs[#]/1*^20, 1])] &),
  PlotLabel -> "net doping (red=n, blue=p)", FrameLabel -> {"x [um]", "y [um]"}];
p2 = ArrayPlot[Reverse[Log10[nAbs + 1]], ColorFunction -> "AvocadoColors", PlotLabel -> "log10 |Nnet|"];
p3 = ListLogPlot[{Table[{x2[[i]], nAbs[[1, i]]}, {i, 1, 160}]}, Joined -> True,
  PlotLabel -> "surface |Nnet|", Frame -> True, FrameLabel -> {"x [um]", "|Nnet|"}];
ixSrc = First@First@Position[Table[Abs[x2[[i]] - 0.05], {i, 1, 160}], Min[Table[Abs[x2[[i]] - 0.05], {i, 1, 160}]]];
ixChan = First@First@Position[Table[Abs[x2[[i]] - 0.40], {i, 1, 160}], Min[Table[Abs[x2[[i]] - 0.40], {i, 1, 160}]]];
p4 = ListLogPlot[{Table[{y2[[j]], nAbs[[j, ixSrc]]}, {j, 1, 140}],
    Table[{y2[[j]], nAbs[[j, ixChan]]}, {j, 1, 140}]}, Joined -> True,
  PlotLabel -> "vertical cuts (src, chan)", Frame -> True, FrameLabel -> {"y [um]", "|Nnet|"},
  PlotLegends -> {"x=0.05", "x=0.40"}];
Export[FileNameJoin[{figDir, "fig2_device_doping.png"}], GraphicsGrid[{{p1, p2}, {p3, p4}}], ImageResolution -> 110];
Print["S3 device: peak |Nnet| = ", ScientificForm[Max[nAbs], 3], " cm^-3"];

(* ---------- fig 3: equilibrium Poisson (90x80) ---------- *)
{x3, y3, X3, Y3} = makeMesh[90, 80];
{Nd3, Na3, Nnet3} = buildDoping[{X3, Y3}];
ps3 = poissonInit[x3, y3, Nnet3, 0.30];
eq3 = poissonSolve[ps3];
Export[FileNameJoin[{figDir, "fig3_poisson_equilibrium.png"}],
  GraphicsGrid[{{ArrayPlot[Reverse[eq3["phi"]], ColorFunction -> "ThermometerColors", PlotLabel -> "equilibrium phi [V]"],
     ArrayPlot[Reverse[Log10[Max[eq3["n"], 1]]], ColorFunction -> "AvocadoColors", PlotLabel -> "log10 n"]},
    {ArrayPlot[Reverse[Log10[Max[eq3["p"], 1]]], ColorFunction -> "SunsetColors", PlotLabel -> "log10 p"]}}],
  ImageResolution -> 110];
Print["S4 Poisson: phi range ", NumberForm[Min[eq3["phi"]], 3], " .. ", NumberForm[Max[eq3["phi"]], 3],
  " V  (", eq3["iter"], " Newton iters, res=", ScientificForm[eq3["res"], 2], ")"];

(* ---------- fig 4: DD 40x34 vs reference ---------- *)
{x4, y4, X4, Y4} = makeMesh[40, 34];
{Nd4, Na4, Nnet4} = buildDoping[{X4, Y4}];
ps4 = poissonInit[x4, y4, Nnet4, 0.30];
dd4 = ddInit[ps4];
t0 = AbsoluteTime[];
res4 = ddSolve[dd4, 0, 3.0, 3.0, 0, 90, 1*^-5];
Print["S5 DD 40x34: ", Round[AbsoluteTime[] - t0], " s, ", res4["iter"], " Gummel iters, dphi=", ScientificForm[res4["dphi"], 2]];
Id4 = ddDrainCurrent[dd4, res4["phi"], res4["n"]];
Print["S5 DD: Id = ", ScientificForm[Id4, 3], " A/um"];
phiM = res4["phi"];
dphi = Abs[phiM - phiRef];
Print["S5 DD verification: max|phi_M - phi_Py| = ", ScientificForm[Max[dphi], 3], " V"];
Export[FileNameJoin[{figDir, "fig4_dd_verify.png"}],
  GraphicsGrid[{{ArrayPlot[Reverse[phiM], ColorFunction -> "ThermometerColors", PlotLabel -> "phi (Mathematica)"],
     ArrayPlot[Reverse[phiRef], ColorFunction -> "ThermometerColors", PlotLabel -> "phi (Python ref)"],
     ArrayPlot[Reverse[dphi], ColorFunction -> "SunsetColors", PlotLabel -> "|diff| [V]"]},
    {ListLinePlot[{Table[{x4[[i]], phiM[[1, i]]}, {i, 1, 40}], Table[{x4[[i]], phiRef[[1, i]]}, {i, 1, 40}]},
      Joined -> True, PlotLabel -> "surface phi (y=0)", PlotLegends -> {"Mathematica", "Python"}]}}],
  ImageResolution -> 110];

(* ---------- fig 5: bulk validation ---------- *)
fields = {0.0, 5*^4, 1*^5, 2*^5, 3*^5};
bulkRes = Table[Module[{ep, f0, Zv, nn, em, te}, {ep, f0} = solveBulk[Ef];
   Zv = Table[dosFn[e], {e, ep}]; nn = trap1D[Zv*f0, ep*eV]; em = trap1D[ep*eV*Zv*f0, ep*eV]/nn;
   te = (2/3)*em/kB; {Ef, ep, f0, te}], {Ef, fields}];
Export[FileNameJoin[{figDir, "fig5_she_bulk.png"}],
  GraphicsGrid[{{
     ListLogPlot[Table[{bulkRes[[k, 2, i]], bulkRes[[k, 3, i]]/bulkRes[[k, 3, 2]]}, {k, 1, 5}, {i, 2, 700}],
      Joined -> True, PlotLabel -> "bulk f0 vs field",
      PlotLegends -> Table["E=" <> ToString[fields[[k]]/1*^3] <> " kV/cm", {k, 1, 5}],
      Frame -> True, FrameLabel -> {"eps [eV]", "f0 (norm)"}, PlotRange -> {{0, 2.0}, {1*^-20, 5}}],
     ListLinePlot[Table[{fields[[k]]/1*^3, bulkRes[[k, 4]]}, {k, 1, 5}], Frame -> True,
      PlotLabel -> "Te vs field", FrameLabel -> {"E [kV/cm]", "Te [K]"}, PlotMarkers -> Automatic]}}],
  ImageResolution -> 110];
Print["S6 bulk: Te = ", Table[NumberForm[bulkRes[[k, 4]], 4], {k, 1, 5}], " K at ",
  Table[NumberForm[fields[[k]]/1*^3, 3], {k, 1, 5}], " kV/cm"];

(* ---------- fig 6-9: SHE 2D 40x34 on the reference DD phi ---------- *)
Print["S7 SHE 40x34 ..."];
t0 = AbsoluteTime[];
she6 = sheInit[x4, y4, phiRef, ctypeRef, neqRef];
Print["S7 SHE: H-grid NH=", she6["NH"], " (Python NH=", Length[HRef], "),  max|H-H_ref| = ",
  ScientificForm[Max[Abs[she6["H"] - HRef]], 2], " J"];
r6 = sheSolve[she6];
Print["S7 SHE total: ", Round[AbsoluteTime[] - t0], " s"];
Print["S7 verification: n_max  M=", ScientificForm[Max[r6["n"]], 5], "  Py=", ScientificForm[Max[nRef], 5], " cm^-3"];
Print["S7 verification: Te_max M=", NumberForm[Max[r6["Te"]], 6], "  Py=", NumberForm[Max[TeRef], 6], " K"];
Print["S7 verification: Gii_max M=", ScientificForm[Max[r6["Gii"]], 5], "  Py=", ScientificForm[Max[GiiRef], 5], " cm^-3/s"];
Print["S7 verification: |vx|max M=", ScientificForm[Max[Abs[r6["vx"]]], 4], "  Py=", ScientificForm[Max[Abs[vxRef]], 4], " cm/s"];
Print["S7 conservation: balance=", ScientificForm[r6["balance"], 3], "  rel=", ScientificForm[r6["relBalance"], 3],
  "  (Python rel: -5.7e-6)"];
Print["S7 currents: Isrc=", ScientificForm[r6["currents"][1], 4], "  Idrn=", ScientificForm[r6["currents"][2], 4],
  " A/um  (Python: -1.704e-4 / 1.707e-4)"];

Export[FileNameJoin[{figDir, "fig6_she_moments.png"}],
  GraphicsGrid[{{ArrayPlot[Reverse[Log10[Max[r6["n"], 1]]], ColorFunction -> "AvocadoColors", PlotLabel -> "log10 n"],
     ArrayPlot[Reverse[r6["Te"]], ColorFunction -> "ThermometerColors", PlotLabel -> "Te [K]"]},
    {ArrayPlot[Reverse[Log10[Max[r6["Gii"], 1]]], ColorFunction -> "SunsetColors", PlotLabel -> "log10 Gii"],
     ArrayPlot[Reverse[Sqrt[r6["vx"]^2 + r6["vy"]^2]], ColorFunction -> "Rainbow", PlotLabel -> "|v| [cm/s]"]}}],
  ImageResolution -> 110];
Export[FileNameJoin[{figDir, "fig7_she_verify_n.png"}],
  GraphicsGrid[{{ArrayPlot[Reverse[Abs[r6["n"] - nRef]/nRef], ColorFunction -> "SunsetColors", PlotLabel -> "rel diff n (M vs Py)"],
     ListLinePlot[{Table[{x4[[i]], Log10[Max[r6["n"][[1, i]], 1]]}, {i, 1, 40}],
       Table[{x4[[i]], Log10[Max[nRef[[1, i]], 1]]}, {i, 1, 40}]},
      Joined -> True, PlotLabel -> "surface log10 n", PlotLegends -> {"M", "Py"}]}}],
  ImageResolution -> 110];
Export[FileNameJoin[{figDir, "fig8_she_verify_Te_Gii.png"}],
  GraphicsGrid[{{ArrayPlot[Reverse[Abs[r6["Te"] - TeRef]], ColorFunction -> "SunsetColors", PlotLabel -> "|dTe| [K]"],
     ArrayPlot[Reverse[Abs[Log10[r6["Gii"] + 1] - Log10[GiiRef + 1]]], ColorFunction -> "SunsetColors", PlotLabel -> "|d log10 Gii|"]},
    {ListLinePlot[{Table[{x4[[i]], r6["Te"][[2, i]]}, {i, 1, 40}], Table[{x4[[i]], TeRef[[2, i]]}, {i, 1, 40}]},
      Joined -> True, PlotLabel -> "Te slice y~0.02um", PlotLegends -> {"M", "Py"}]}}],
  ImageResolution -> 110];
cj = 2; ci = 21;
F3r = BinaryReadList[FileNameJoin[{refDir, "F3d_ref.bin"}], "Real32"];
F3r = ArrayReshape[F3r, {34, 40, 565}];
idxH = Flatten[Position[she6["active"][[cj, ci, All]], 1]];
Export[FileNameJoin[{figDir, "fig9_f0_slice.png"}],
  ListLogPlot[{Table[{she6["H"][[kk]]/eV, Max[r6["F3"][[cj, ci, kk]], 1*^-300]}, {kk, idxH}],
    Table[{she6["H"][[kk]]/eV, Max[F3r[[cj, ci, kk]], 1*^-300]}, {kk, idxH}]},
   Joined -> True, PlotLabel -> "f0(H) at (j=2,i=21)", PlotLegends -> {"Mathematica", "Python"},
   Frame -> True, FrameLabel -> {"H [eV]", "f0"}, PlotRange -> All], ImageResolution -> 110];

(* ---------- verification summary ---------- *)
summary = {
  {"quantity", "Mathematica", "Python ref", "diff"},
  {"n_max [cm^-3]", ToString[ScientificForm[Max[r6["n"]], 5]], ToString[ScientificForm[Max[nRef], 5]],
   ToString[ScientificForm[Abs[Max[r6["n"]] - Max[nRef]]/Max[nRef], 2]]},
  {"Te_max [K]", ToString[NumberForm[Max[r6["Te"]], 6]], ToString[NumberForm[Max[TeRef], 6]],
   ToString[NumberForm[Abs[Max[r6["Te"]] - Max[TeRef]], 4]]},
  {"Gii_max [cm^-3/s]", ToString[ScientificForm[Max[r6["Gii"]], 5]], ToString[ScientificForm[Max[GiiRef], 5]],
   ToString[ScientificForm[Abs[Max[r6["Gii"]] - Max[GiiRef]]/Max[GiiRef], 2]]},
  {"|vx|max [cm/s]", ToString[ScientificForm[Max[Abs[r6["vx"]]], 4]], ToString[ScientificForm[Max[Abs[vxRef]], 4]],
   ToString[ScientificForm[Abs[Max[Abs[r6["vx"]]] - Max[Abs[vxRef]]]/Max[Abs[vxRef]], 2]]},
  {"phi max|diff| [V]", "-", "-", ToString[ScientificForm[Max[Abs[phiM - phiRef]], 3]]},
  {"Idrain [A/um]", ToString[ScientificForm[r6["currents"][2], 4]], "1.707e-4",
   ToString[ScientificForm[Abs[r6["currents"][2] - 1.707*^-4]/1.707*^-4, 2]]},
  {"number balance (rel)", ToString[ScientificForm[r6["relBalance"], 2]], "-5.7e-6", "-"}
};
Export[FileNameJoin[{dir, "verification_summary.txt"}], TableForm[summary], "Text"];
Print["=== VERIFICATION SUMMARY ==="];
Print[TableForm[summary]];
Print["Done. Figures in ", figDir];
