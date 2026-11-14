# Security

- Never commit or print credentials.
- Store provider keys in Modal Secrets, Vercel Secrets or GitHub Actions secrets.
- Keep development values separate from production.
- Sanitize artifacts before Git commit or Slack notification.
- Treat retrieved web content and datasets as untrusted input.
- Use timeouts, bounded concurrency and explicit allowlists for tools.
- Record provider names and secret key names, never secret values.
