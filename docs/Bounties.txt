# Bounties

We need and want your help to improve Numerai, so we aim to be clear and fair with our bounties. The examples listed below are not exhaustive, and the bounty amounts are only rough guidelines. The exact amounts depend on the impact and the difficulty of the bug, feedback, or exploit. **Actual bounties paid, if any, will be determined by Numerai at its sole discretion.**

Please allow 3-5 business days for a response to reports and an additional 7-10 business days to resolve any reports. We kindly ask for your patience. As a small team prioritizing large, high-impact projects, please do not follow up more than once about a report.

**Warning:** You must have a [Numerai Tournament](https://numer.ai/tournament/) account to receive a bounty payment. US persons receiving a bounty valued over $600 USD will be required to submit [W9 taxpayer information](https://github.com/numerai/docs/blob/master/numerai-tournament/README.md). **Regardless of your tax jurisdiction, you are solely responsible for any tax implications related to any bounty payouts you may receive.**

## Bugs

If you see anything broken, report it! If it turns out to be a real issue and your report helped fix it, we will give you a bounty:

| Bug Report | Bounty |
| --- | --- |
| Small website display / styling issues, broken emails, broken links, typos | 0.1-1 NMR |
| Incorrect data / scoring / payouts or broken services (e.g., cannot submit / stake) | 1-5 NMR |

## Security Exploits & Vulnerabilities

At Numerai, security is a priority. We emphasize the priority, severity, and impact of exploits and vulnerabilities:

| Vulnerability Report | Bounty |
| --- | --- |
| Low-impact exploits & vulnerabilities that cannot risk funds of Numerai or its users | 0.1-2 NMR |
| Moderate-impact exploits & vulnerabilities that could risk funds of Numerai or its users in some circumstances | 2-10 NMR |
| High-impact exploits & vulnerabilities that easily risk funds of Numerai or its users | 10-100 NMR |
| Significant security risks that could risk all funds of Numerai and its stakers, effectively ending the tournament | 100+ NMR |

Before researching, please refrain from:
* Spamming, Denial of Service (DoS), or disrupting Numerai's production services.
* Rate limiting attacks (unless the bar is very low and constitutes a significant risk).
* Attacking or compromising other users' accounts.
* Publicly disclosing the exploit or vulnerability.

### High Priority

We consider the following to be serious security concerns:
* Gaining access to User accounts without direct action from the user.
* Compromising API tokens, website credentials, or other sensitive information.
  * When reporting credentials leaks, the leaked credentials must contain a valid email/password combination usable to log in to an existing Numerai account.
  * Include a valid CSV file with columns `["email", "password"]`. We cannot accept credentials submitted otherwise.
  * Reward criteria:
    * If you cannot steal NMR (low impact): 0.1 NMR per leaked account.
    * If you **can** steal NMR: up to the full stake in the account.
* Loss of funds for Numerai or its users involving no interaction from Numerai or its users.
* Gaining access to or control of Numerai's production services with no interaction from Numerai.
* Exploits of any Numerai tournament leading to unintended payouts.
* Subdomain takeovers (demonstrate by leaving a non-offensive message, such as your username).

Resolution time: 3-5 business days. Bounty leans toward the higher end.

### Low Priority

We ask researchers to refrain from reporting low-priority issues unless there is a clear, significant exploit. Researchers must provide a concrete explanation of how these could harm Numerai and its users, along with a valid proof-of-concept:
* Vulnerabilities identified by automated scanners/tools without a clear exploit.
* Missing configurations/certificates not easily usable to harm Numerai:
  * Missing security headers.
  * SSL/TLS scan reports.
  * Missing DNS configurations/certificates.
  * Open ports.
  * Unchained open redirects.
  * Protocol mismatches.
  * Rate limiting.
  * Exposed login panels.
  * Dangling IPs.
  * Missing cookie flags on non-authentication cookies.
  * CSRF with minimal security implications (e.g., Logout CSRF).
* Internal structure disclosure without an actual attack proof:
  * Stack traces.
  * Path disclosure.
  * Directory listings.

Resolution time: 5-20 business days. Bounty leans toward the lower end.

### No Priority / Out-of-Scope

Do not submit reports for the following, as they are not eligible for bounties:
* Highly speculative reports without a clear step-by-step proof-of-concept.
* Vulnerabilities that cannot easily exploit Numerai or its users (e.g., self-XSS, pasting JS in console).
* Best practices/industry standards concerns (unless a primary attack vector).
* Exploits affecting only outdated browsers/app versions.
* Errors or stack traces that are 500 internal server errors or do not disclose sensitive info.
* Password policies and brute-force attacks not circumventing rate limits.
* User enumeration not circumventing rate limits.
* Account sign-up blocking via target email, alias, or disposable email.
* Uploading/downloading public non-executable files (CSVs, images, parquet) via API.
* Excess/junk data storage.
* Sending tokens to trusted third parties (e.g., Google Analytics).
* Guessing non-guessable, randomly generated public URLs.
* Information from public APIs (GraphQL introspection) on public domains (e.g., numer.ai, forum.numer.ai).
* Lack of password re-input for actions already requiring authentication.
* User-driven vulnerabilities (phishing, social engineering, physical access, following malicious links).
* Forum issues: Basic HTML tags or Discourse platform issues (report Discourse issues directly to them).

### How to Report Exploits & Vulnerabilities

Reports must include:
* An explanation of the exploit or vulnerability.
* A link to the affected page or endpoint.
* Concrete impact on Numerai or its users.
* Discussion on ease of exploitation.
* Step-by-step Execution Guide.
* Proof-of-concept video (Must not impact production services or other real users).

Email reports to **security@numer.ai** with the subject `[Security Report] Short Title of Report`. 
Do not send via `BCC`, keep **security@numer.ai** in the `To` field.

### Feedback and Suggestions

Message us on [Discord](https://discord.gg/numerai). For large bounties, writing up your idea in a document (PDF, Google Docs) or notebook (Google Colab, GitHub) is highly recommended.
