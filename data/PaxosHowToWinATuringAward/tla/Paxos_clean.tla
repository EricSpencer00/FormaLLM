-------------------------------- MODULE Paxos -------------------------------

EXTENDS Integers

CONSTANTS Value, Acceptor, Quorum

ASSUME  /\ \A Q \in Quorum : Q \subseteq Acceptor
/\ \A Q1, Q2 \in Quorum : Q1 \cap Q2 /= {}

Ballot ==  Nat

None == CHOOSE v : v \notin Ballot

Message ==
[type : {"1a"}, bal : Ballot]
\cup [type : {"1b"}, acc : Acceptor, bal : Ballot,
mbal : Ballot \cup {-1}, mval : Value \cup {None}]
\cup [type : {"2a"}, bal : Ballot, val : Value]
\cup [type : {"2b"}, acc : Acceptor, bal : Ballot, val : Value]
-----------------------------------------------------------------------------

VARIABLES maxBal, maxVBal, maxVal, msgs
vars == <<maxBal, maxVBal, maxVal, msgs>>

TypeOK == /\ maxBal  \in [Acceptor -> Ballot \cup {-1}]
/\ maxVBal \in [Acceptor -> Ballot \cup {-1}]
/\ maxVal  \in [Acceptor -> Value \cup {None}]
/\ msgs \subseteq Message

Init == /\ maxBal  = [a \in Acceptor |-> -1]
/\ maxVBal = [a \in Acceptor |-> -1]
/\ maxVal  = [a \in Acceptor |-> None]
/\ msgs = {}
----------------------------------------------------------------------------

Send(m) == msgs' = msgs \cup {m}

Phase1a(b) == /\ Send([type |-> "1a", bal |-> b])
/\ UNCHANGED <<maxBal, maxVBal, maxVal>>

Phase1b(a) ==
/\ \E m \in msgs :
/\ m.type = "1a"
/\ m.bal > maxBal[a]
/\ maxBal' = [maxBal EXCEPT ![a] = m.bal]
/\ Send([type |-> "1b", acc |-> a, bal |-> m.bal,
mbal |-> maxVBal[a], mval |-> maxVal[a]])
/\ UNCHANGED <<maxVBal, maxVal>>

Phase2a(b, v) ==
/\ ~ \E m \in msgs : m.type = "2a" /\ m.bal = b
/\ \E Q \in Quorum :
LET Q1b == {m \in msgs : /\ m.type = "1b"
/\ m.acc \in Q
/\ m.bal = b}
Q1bv == {m \in Q1b : m.mbal >= 0}
IN  /\ \A a \in Q : \E m \in Q1b : m.acc = a
/\ \/ Q1bv = {}
\/ \E m \in Q1bv :
/\ m.mval = v
/\ \A mm \in Q1bv : m.mbal >= mm.mbal
/\ Send([type |-> "2a", bal |-> b, val |-> v])
/\ UNCHANGED <<maxBal, maxVBal, maxVal>>

Phase2b(a) ==
\E m \in msgs :
/\ m.type = "2a"
/\ m.bal >= maxBal[a]
/\ maxBal' = [maxBal EXCEPT ![a] = m.bal]
/\ maxVBal' = [maxVBal EXCEPT ![a] = m.bal]
/\ maxVal' = [maxVal EXCEPT ![a] = m.val]
/\ Send([type |-> "2b", acc |-> a,
bal |-> m.bal, val |-> m.val])

Next == \/ \E b \in Ballot : \/ Phase1a(b)
\/ \E v \in Value : Phase2a(b, v)
\/ \E a \in Acceptor : Phase1b(a) \/ Phase2b(a)

Spec == Init /\ [][Next]_vars
----------------------------------------------------------------------------

votes ==
[a \in Acceptor |->
{<<m.bal, m.val>> : m \in {mm \in msgs: /\ mm.type = "2b"
/\ mm.acc = a }}]

V == INSTANCE Voting

Inv ==
/\ TypeOK
/\ \A a \in Acceptor : maxBal[a] >= maxVBal[a]
/\ \A a \in Acceptor : IF maxVBal[a] = -1
THEN maxVal[a] = None
ELSE <<maxVBal[a], maxVal[a]>> \in votes[a]
/\ \A m \in msgs :
/\ (m.type = "1b") => /\ maxBal[m.acc] >= m.bal
/\ (m.mbal >= 0) =>
<<m.mbal, m.mval>> \in votes[m.acc]
/\ (m.type = "2a") => /\ \E Q \in Quorum :
V!ShowsSafeAt(Q, m.bal, m.val)
/\ \A mm \in msgs : /\ mm.type ="2a"
/\ mm.bal = m.bal
=> mm.val = m.val
/\ (m.type = "2b") => /\ maxVBal[m.acc] >= m.bal
/\ \E mm \in msgs : /\ mm.type = "2a"
/\ mm.bal  = m.bal
/\ mm.val  = m.val

THEOREM Invariance  ==  Spec => []Inv

THEOREM Implementation  ==  Spec => V!Spec

============================================================================
