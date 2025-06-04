----------------------------- MODULE Disruptor_SPMC ------------------------

EXTENDS Integers, FiniteSets, Sequences

CONSTANTS
MaxPublished,
Writers,
Readers,
Size,
NULL

ASSUME Writers /= {}
ASSUME Readers /= {}
ASSUME Size         \in Nat \ {0}
ASSUME MaxPublished \in Nat \ {0}

VARIABLES
ringbuffer,
published,
read,
pc,
consumed

vars == <<
ringbuffer,
published,
read,
consumed,
pc
>>

Access  == "Access"
Advance == "Advance"

Transition(t, from, to) ==
/\ pc[t] = from
/\ pc'   = [ pc EXCEPT ![t] = to ]

Buffer == INSTANCE RingBuffer WITH Values <- Int

Range(f) ==
{ f[x] : x \in DOMAIN(f) }

MinReadSequence ==
CHOOSE min \in Range(read) : \A r \in Readers : min <= read[r]

BeginWrite(writer) ==
LET
next     == published + 1
index    == Buffer!IndexOf(next)
min_read == MinReadSequence
IN

/\ min_read >= next - Size
/\ Transition(writer, Advance, Access)
/\ Buffer!Write(index, writer, next)
/\ UNCHANGED << consumed, published, read >>

EndWrite(writer) ==
LET
next  == published + 1
index == Buffer!IndexOf(next)
IN
/\ Transition(writer, Access, Advance)
/\ Buffer!EndWrite(index, writer)
/\ published' = next
/\ UNCHANGED << consumed, read >>

BeginRead(reader) ==
LET
next  == read[reader] + 1
index == Buffer!IndexOf(next)
IN
/\ published >= next
/\ Transition(reader, Advance, Access)
/\ Buffer!BeginRead(index, reader)

/\ consumed' = [ consumed EXCEPT ![reader] = Append(@, Buffer!Read(index)) ]
/\ UNCHANGED << published, read >>

EndRead(reader) ==
LET
next  == read[reader] + 1
index == Buffer!IndexOf(next)
IN
/\ Transition(reader, Access, Advance)
/\ Buffer!EndRead(index, reader)
/\ read' = [ read EXCEPT ![reader] = next ]
/\ UNCHANGED << consumed, published >>

Init ==
/\ Buffer!Init
/\ published = -1
/\ read      = [ r \in Readers                |-> -1      ]
/\ consumed  = [ r \in Readers                |-> << >>   ]
/\ pc        = [ a \in Writers \union Readers |-> Advance ]

Next ==
\/ \E w \in Writers : BeginWrite(w)
\/ \E w \in Writers : EndWrite(w)
\/ \E r \in Readers : BeginRead(r)
\/ \E r \in Readers : EndRead(r)

Fairness ==
/\ \A r \in Readers : WF_vars(BeginRead(r))
/\ \A r \in Readers : WF_vars(EndRead(r))

Spec ==
Init /\ [][Next]_vars /\ Fairness

StateConstraint == published < MaxPublished

TypeOk ==
/\ Buffer!TypeOk
/\ published \in Int
/\ read      \in [ Readers                -> Int                 ]
/\ consumed  \in [ Readers                -> Seq(Nat)            ]
/\ pc        \in [ Writers \union Readers -> { Access, Advance } ]

NoDataRaces == Buffer!NoDataRaces

Liveliness ==
\A r \in Readers : \A i \in 0 .. (MaxPublished - 1) :
<>[](i \in 0 .. published => Len(consumed[r]) >= i + 1 /\ consumed[r][i + 1] = i)

=============================================================================
