---------------------- MODULE AllocatorImplementation -------------------

EXTENDS FiniteSets, Sequences, Naturals

CONSTANTS
Clients,
Resources

ASSUME
IsFiniteSet(Resources)

VARIABLES

unsat,
alloc,
sched,

requests,
holding,

network

Sched == INSTANCE SchedulingAllocator

-------------------------------------------------------------------------

Messages ==
[type : {"request", "allocate", "return"},
clt : Clients,
rsrc : SUBSET Resources]

TypeInvariant ==
/\ Sched!TypeInvariant
/\ requests \in [Clients -> SUBSET Resources]
/\ holding \in [Clients -> SUBSET Resources]
/\ network \in SUBSET Messages

-------------------------------------------------------------------------

Init ==
/\ Sched!Init
/\ requests = [c \in Clients |-> {}]
/\ holding = [c \in Clients |-> {}]
/\ network = {}

Request(c,S) ==
/\ requests[c] = {} /\ holding[c] = {}
/\ S # {} /\ requests' = [requests EXCEPT ![c] = S]
/\ network' = network \cup {[type |-> "request", clt |-> c, rsrc |-> S]}
/\ UNCHANGED <<unsat,alloc,sched,holding>>

RReq(m) ==
/\ m \in network /\ m.type = "request"
/\ alloc[m.clt] = {}
/\ unsat' = [unsat EXCEPT ![m.clt] = m.rsrc]
/\ network' = network \ {m}
/\ UNCHANGED <<alloc,sched,requests,holding>>

Allocate(c,S) ==
/\ Sched!Allocate(c,S)
/\ network' = network \cup {[type |-> "allocate", clt |-> c, rsrc |-> S]}
/\ UNCHANGED <<requests,holding>>

RAlloc(m) ==
/\ m \in network /\ m.type = "allocate"
/\ holding' = [holding EXCEPT ![m.clt] = @ \cup m.rsrc]
/\ requests' = [requests EXCEPT ![m.clt] = @ \ m.rsrc]
/\ network' = network \ {m}
/\ UNCHANGED <<unsat,alloc,sched>>

Return(c,S) ==
/\ S # {} /\ S \subseteq holding[c]
/\ holding' = [holding EXCEPT ![c] = @ \ S]
/\ network' = network \cup {[type |-> "return", clt |-> c, rsrc |-> S]}
/\ UNCHANGED <<unsat,alloc,sched,requests>>

RRet(m) ==
/\ m \in network /\ m.type = "return"
/\ alloc' = [alloc EXCEPT ![m.clt] = @ \ m.rsrc]
/\ network' = network \ {m}
/\ UNCHANGED <<unsat,sched,requests,holding>>

Schedule ==
/\ Sched!Schedule
/\ UNCHANGED <<requests,holding,network>>

Next ==
\/ \E c \in Clients, S \in SUBSET Resources :
Request(c,S) \/ Allocate(c,S) \/ Return(c,S)
\/ \E m \in network :
RReq(m) \/ RAlloc(m) \/ RRet(m)
\/ Schedule

vars == <<unsat,alloc,sched,requests,holding,network>>

-------------------------------------------------------------------------

Liveness ==
/\ \A c \in Clients : WF_vars(requests[c]={} /\ Return(c,holding[c]))
/\ \A c \in Clients : WF_vars(\E S \in SUBSET Resources : Allocate(c, S))
/\ WF_vars(Schedule)
/\ \A m \in Messages :
/\ WF_vars(RReq(m))
/\ WF_vars(RAlloc(m))
/\ WF_vars(RRet(m))

Specification == Init /\ [][Next]_vars /\ Liveness

-------------------------------------------------------------------------

RequestsInTransit(c) ==
{ msg.rsrc : msg \in {m \in network : m.type = "request" /\ m.clt = c} }

AllocsInTransit(c) ==
{ msg.rsrc : msg \in {m \in network : m.type = "allocate" /\ m.clt = c} }

ReturnsInTransit(c) ==
{ msg.rsrc : msg \in {m \in network : m.type = "return" /\ m.clt = c} }

Invariant ==

/\ Sched!AllocatorInvariant

/\ \A c \in Clients :
/\ Cardinality(RequestsInTransit(c)) <= 1
/\ requests[c] = unsat[c]
\cup UNION RequestsInTransit(c)
\cup UNION AllocsInTransit(c)
/\ alloc[c] = holding[c]
\cup UNION AllocsInTransit(c)
\cup UNION ReturnsInTransit(c)

ResourceMutex ==
\A c1,c2 \in Clients : holding[c1] \cap holding[c2] # {} => c1 = c2

ClientsWillReturn ==
\A c \in Clients: (requests[c]={} ~> holding[c]={})

ClientsWillObtain ==
\A c \in Clients, r \in Resources : r \in requests[c] ~> r \in holding[c]

InfOftenSatisfied ==
\A c \in Clients : []<>(requests[c] = {})

-------------------------------------------------------------------------

THEOREM Specification => []TypeInvariant
THEOREM Specification => []ResourceMutex
THEOREM Specification => []Invariant
THEOREM Specification => ClientsWillReturn
THEOREM Specification => ClientsWillObtain
THEOREM Specification => InfOftenSatisfied

-------------------------------------------------------------------------

SchedAllocator == Sched!Allocator

THEOREM Specification => SchedAllocator
=========================================================================
