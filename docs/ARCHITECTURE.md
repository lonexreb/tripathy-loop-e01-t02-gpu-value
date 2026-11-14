# Architecture

Use ports and adapters so research logic is independent of Exa, past.dev, Modal, Vercel, GitHub, Linear and Slack.

- `domain`: hypotheses, evidence, predictions and evaluations.
- `application`: experiment orchestration and stop conditions.
- `adapters`: search, memory, compute, storage and notifications.
- `presentation`: CLI, dashboard or demo.

Do not introduce an adapter until the idea requires it. Every adapter must have a narrow interface, timeout, typed failure and test substitute.
