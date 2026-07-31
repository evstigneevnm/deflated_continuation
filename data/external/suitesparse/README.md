# SuiteSparse Matrix Collection fixtures

These files are the report-validation inputs named in
`/home/noctum/Documents/LaTex/Eigs_Paper/report_EIGS.tex`.

## Sources

| Matrix | Collection page | Original Matrix Market archive |
| --- | --- | --- |
| `Bai/rdb450` | <https://sparse.tamu.edu/Bai/rdb450> | <https://sparse.tamu.edu/MM/Bai/rdb450.tar.gz> |
| `Rommes/S80PI_n1` | <https://sparse.tamu.edu/Rommes/S80PI_n1> | <https://sparse.tamu.edu/MM/Rommes/S80PI_n1.tar.gz> |

The archives were downloaded on 2026-07-25. Their SHA-256 digests were:

```text
1116f378b96ad63bdf81db8ab7868549215c3723948bc6c3898e5091f8719878  rdb450.tar.gz
76819fe3b0a0f092171e3700defb651b7ccb83a2ecbb473656b6c1196f1f84d4  S80PI_n1.tar.gz
```

Only the matrices needed by the report tests are retained. `S80PI_n1_B.mtx`
and `S80PI_n1_C.mtx` are not required for eigenvalue validation.

## S80PI_n1 interpretation

The authoritative collection matrix `S80PI_n1.mtx` is 4028 by 4028 with
9927 stored entries and rank 4025. The report calls it a 4025 by 4025
matrix; that number agrees with the rank, not the stored dimensions.

The collection also supplies the positive diagonal matrix
`S80PI_n1_E.mtx` for the descriptor model

```text
E dx/dt = A x + B u.
```

The report treats `S80PI_n1` as the standard matrix `A`, quotes its rank as
its dimension, and ignores the three-dimensional nullspace. This
interpretation is also consistent with the quoted frequencies near
`+/-1.87i` and `+/-1.68i`; they occur in the spectrum of `A`. Stage-3 tests
therefore preserve both `A` and `E`, but reproduce the report against `A`.

See `SHA256SUMS` for the retained-file digests.
