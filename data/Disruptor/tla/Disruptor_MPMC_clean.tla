--------------------------- MODULE Disruptor_MPMC --------------------------

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
next_sequence,
claimed_sequence,
published,
read,
pc,
consumed

vars == <<
ringbuffer,
next_sequence,
claimed_sequence,
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

Xor(A, B) == A = ~B

Range(f) ==
{ f[x] : x \in DOMAIN(f) }

MinReadSequence ==
CHOOSE min \in Range(read) :
\A r \in Readers : min <= read[r]

MinClaimedSequence ==
CHOOSE min \in Range(claimed_sequence) :
\A w \in Writers : min <= claimed_sequence[w]

IsPublished(sequence) ==
LET
index   == Buffer!IndexOf(sequence)

round   == sequence \div Size
is_even == round % 2 = 0
IN

published[index] = is_even

Publish(sequence) ==
LET index == Buffer!IndexOf(sequence)

IN  published' = [ published EXCEPT ![index] = Xor(TRUE, @) ]

AvailablePublishedSequence ==
LET guaranteed_published == MinClaimedSequence - 1
candidate_sequences  == {guaranteed_published} \union Range(claimed_sequence)
IN  CHOOSE max \in candidate_sequences :
IsPublished(max) => ~ \E w \in Writers :
/\ claimed_sequence[w] = max + 1
/\ IsPublished(claimed_sequence[w])

BeginWrite(writer) ==
LET
seq      == next_sequence
index    == Buffer!IndexOf(seq)
min_read == MinReadSequence
IN

/\ min_read >= seq - Size
/\ claimed_sequence' = [ claimed_sequence EXCEPT ![writer] = seq ]
/\ next_sequence'    = seq + 1
/\ Transition(writer, Advance, Access)
/\ Buffer!Write(index, writer, seq)
/\ UNCHANGED << consumed, published, read >>

EndWrite(writer) ==
LET
seq   == claimed_sequence[writer]
index == Buffer!IndexOf(seq)
IN
/\ Transition(writer, Access, Advance)
/\ Buffer!EndWrite(index, writer)
/\ Publish(seq)
/\ UNCHANGED << claimed_sequence, next_sequence, consumed, read >>

BeginRead(reader) ==
LET
next  == read[reader] + 1
index == Buffer!IndexOf(next)
IN
/\ IsPublished(next)
/\ Transition(reader, Advance, Access)
/\ Buffer!BeginRead(index, reader)

/\ consumed' = [ consumed EXCEPT ![reader] = Append(@, Buffer!Read(index)) ]
/\ UNCHANGED << claimed_sequence, next_sequence, published, read >>

EndRead(reader) ==
LET
next  == read[reader] + 1
index == Buffer!IndexOf(next)
IN
/\ Transition(reader, Access, Advance)
/\ Buffer!EndRead(index, reader)
/\ read' = [ read EXCEPT ![reader] = next ]
/\ UNCHANGED << claimed_sequence, next_sequence, consumed, published >>

Init ==
/\ Buffer!Init
/\ next_sequence    = 0
/\ claimed_sequence = [ w \in Writers                |-> -1      ]
/\ published        = [ i \in 0..Buffer!LastIndex    |-> FALSE   ]
/\ read             = [ r \in Readers                |-> -1      ]
/\ consumed         = [ r \in Readers                |-> << >>   ]
/\ pc               = [ a \in Writers \union Readers |-> Advance ]

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

StateConstraint == next_sequence <= MaxPublished

NoDataRaces == Buffer!NoDataRaces

TypeOk ==
/\ Buffer!TypeOk
/\ next_sequence    \in Nat
/\ claimed_sequence \in [ Writers                -> Int                 ]
/\ published        \in [ 0..Buffer!LastIndex    -> { TRUE, FALSE }     ]
/\ read             \in [ Readers                -> Int                 ]
/\ consumed         \in [ Readers                -> Seq(Nat)            ]
/\ pc               \in [ Writers \union Readers -> { Access, Advance } ]

Liveliness ==
\A r \in Readers : \A i \in 0..(MaxPublished - 1) :
<>[](i \in 0..AvailablePublishedSequence => Len(consumed[r]) >= i + 1 /\ consumed[r][i + 1] = i)

=============================================================================
