----------------------------- MODULE RingBuffer -----------------------------

LOCAL INSTANCE Naturals
LOCAL INSTANCE FiniteSets

CONSTANTS
Size,
Readers,
Writers,
Values,
NULL

ASSUME Size \in Nat \ {0}
ASSUME Writers /= {}
ASSUME Readers /= {}
ASSUME NULL \notin Values

VARIABLE ringbuffer

LastIndex == Size - 1

NoDataRaces ==
\A i \in 0 .. LastIndex :
/\ ringbuffer.readers[i] = {} \/ ringbuffer.writers[i] = {}
/\ Cardinality(ringbuffer.writers[i]) <= 1

Init ==
ringbuffer = [
slots   |-> [i \in 0 .. LastIndex |-> NULL ],
readers |-> [i \in 0 .. LastIndex |-> {}   ],
writers |-> [i \in 0 .. LastIndex |-> {}   ]
]

IndexOf(sequence) ==
sequence % Size

Write(index, writer, value) ==
ringbuffer' = [
ringbuffer EXCEPT
!.writers[index] = @ \union { writer },
!.slots[index]   = value
]

EndWrite(index, writer) ==
ringbuffer' = [ ringbuffer EXCEPT !.writers[index] = @ \ { writer } ]

BeginRead(index, reader) ==
ringbuffer' = [ ringbuffer EXCEPT !.readers[index] = @ \union { reader } ]

Read(index) ==
ringbuffer.slots[index]

EndRead(index, reader) ==
ringbuffer' = [ ringbuffer EXCEPT !.readers[index] = @ \ { reader } ]

TypeOk ==
ringbuffer \in [
slots:   UNION { [0 .. LastIndex -> Values \union { NULL }] },
readers: UNION { [0 .. LastIndex -> SUBSET(Readers)       ] },
writers: UNION { [0 .. LastIndex -> SUBSET(Writers)       ] }
]

=============================================================================
