------------------------------- MODULE bosco -------------------------------

EXTENDS Naturals, FiniteSets

CONSTANTS N, T, F

moreNplus3Tdiv2 == (N + 3 * T) \div 2 + 1
moreNminusTdiv2 == (N - T) \div 2 + 1

VARIABLE pc, rcvd, sent

ASSUME  /\ N \in Nat /\ T \in Nat /\ F \in Nat
/\ moreNplus3Tdiv2 \in Nat /\ moreNminusTdiv2 \in Nat

ASSUME (N > 3 * T) /\ (T >= F) /\ (F >= 0)

ASSUME  \/ 2 * moreNplus3Tdiv2 = N + 3 * T + 1
\/ 2 * moreNplus3Tdiv2 = N + 3 * T + 2

ASSUME  \/ 2 * moreNminusTdiv2 = N - T + 1
\/ 2 * moreNminusTdiv2 = N - T + 2

P == 1 .. N
Corr == 1 .. N - F
Faulty == N - F + 1 .. N

M == { "ECHO0", "ECHO1" }

vars == << pc, rcvd, sent >>

rcvd0(self) == Cardinality({m \in rcvd'[self] : m[2] = "ECHO0"})

rcvd1(self) == Cardinality({m \in rcvd'[self] : m[2] = "ECHO1"})

rcvd01(self) == Cardinality({p \in P : (\E m \in rcvd'[self] : m[1] = p)})

Receive(self) ==
\E r \in SUBSET (P \times M):
/\ r \subseteq (sent \cup { <<p, "ECHO0">> : p \in Faulty }
\cup { <<p, "ECHO1">> : p \in Faulty })
/\ rcvd[self] \subseteq r
/\ rcvd' = [rcvd EXCEPT ![self] = r ]

UponV0(self) ==
/\ pc[self] = "V0"
/\ sent' = sent \cup { <<self, "ECHO0">> }
/\ pc' = [pc EXCEPT ![self] = "S0"]

UponV1(self) ==
/\ pc[self] = "V1"
/\ pc' = [pc EXCEPT ![self] = "S1"]
/\ sent' = sent \cup { <<self, "ECHO1">> }

UponOneStep0(self) ==
/\ pc[self] = "S0" \/ pc[self] = "S1"
/\ rcvd01(self) >= N - T
/\ rcvd0(self) >= moreNplus3Tdiv2
/\ pc' = [pc EXCEPT ![self] = "D0"]
/\ sent' = sent

UponOneStep1(self) ==
/\ pc[self] = "S0" \/ pc[self] = "S1"
/\ rcvd01(self) >= N - T
/\ rcvd1(self) >= moreNplus3Tdiv2
/\ pc' = [pc EXCEPT ![self] = "D1"]
/\ sent' = sent

UponUnderlying0(self) ==
/\ pc[self] = "S0" \/ pc[self] = "S1"
/\ rcvd01(self) >= N - T
/\ rcvd0(self) >= moreNminusTdiv2
/\ rcvd1(self) < moreNminusTdiv2
/\ pc' = [pc EXCEPT ![self] = "U0"]
/\ sent' = sent
/\ rcvd0(self) < moreNplus3Tdiv2
/\ rcvd1(self) < moreNplus3Tdiv2

UponUnderlying1(self) ==
/\ pc[self] = "S0" \/ pc[self] = "S1"
/\ rcvd01(self) >= N - T
/\ rcvd1(self) >= moreNminusTdiv2
/\ rcvd0(self) < moreNminusTdiv2
/\ pc' = [pc EXCEPT ![self] = "U1"]
/\ sent' = sent
/\ rcvd0(self) < moreNplus3Tdiv2
/\ rcvd1(self) < moreNplus3Tdiv2

UponUnderlyingUndecided(self) ==
/\ pc[self] = "S0" \/ pc[self] = "S1"
/\ rcvd01(self) >= N - T
/\ rcvd0(self) >= moreNminusTdiv2
/\ rcvd1(self) >= moreNminusTdiv2
/\  \/ pc[self] = "S0" /\ pc' = [pc EXCEPT ![self] = "U0"]
\/ pc[self] = "S1" /\ pc' = [pc EXCEPT ![self] = "U1"]
/\ sent' = sent
/\ rcvd0(self) < moreNplus3Tdiv2
/\ rcvd1(self) < moreNplus3Tdiv2

Step(self) ==
/\ Receive(self)
/\ \/ UponV0(self)
\/ UponV1(self)
\/ UponOneStep0(self)
\/ UponOneStep1(self)
\/ UponUnderlying0(self)
\/ UponUnderlying1(self)
\/ UponUnderlyingUndecided(self)
\/ pc' = pc /\ sent' = sent

Init ==
/\ pc \in [ Corr -> {"V0", "V1"} ]
/\ sent = {}
/\ rcvd = [ i \in Corr |-> {} ]

Next ==  (\E self \in Corr: Step(self))

Spec == Init /\ [][Next]_vars

TypeOK ==
/\ sent \subseteq P \times M
/\ pc \in [ Corr -> {"V0", "V1", "S0", "S1", "D0", "D1", "U0", "U1"} ]
/\ rcvd \in [ Corr -> SUBSET (P \times M) ]

Lemma3_0 == (\E self \in Corr: (pc[self] = "D0")) => (\A self \in Corr: (pc[self] /= "D1"))
Lemma3_1 == (\E self \in Corr: (pc[self] = "D1")) => (\A self \in Corr: (pc[self] /= "D0"))

Lemma4_0 == (\E self \in Corr: (pc[self] = "D0")) => (\A self \in Corr: (pc[self] /= "U1"))
Lemma4_1 == (\E self \in Corr: (pc[self] = "D1")) => (\A self \in Corr: (pc[self] /= "U0"))

OneStep0 ==
(\A i \in Corr: pc[i] = "V0")
=> [](\A i \in Corr:
/\ pc[i] /= "D1"
/\ pc[i] /= "U0"
/\ pc[i] /= "U1")

OneStep1 ==
(\A i \in Corr: pc[i] = "V1")
=> [](\A i \in Corr:
/\ pc[i] /= "D0"
/\ pc[i] /= "U0"
/\ pc[i] /= "U1")

=============================================================================
