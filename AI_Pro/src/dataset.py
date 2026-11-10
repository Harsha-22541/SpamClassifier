"""Dataset loading, splitting, and benchmark generation utilities."""

import csv
import os
import random
from typing import List, Tuple, Dict, Any, Optional

# Realistic curated email benchmark examples for instant testing and demonstration
CURATED_BENCHMARK_EMAILS: List[Dict[str, str]] = [
    {
        "id": "spam_urgent_account",
        "subject": "URGENT: Your Account Has Been Suspended - Verify Immediately",
        "body": "Dear Customer, We detected unauthorized login attempts from an unknown device. Your access has been restricted. Click the secure link below to verify your password and identity within 24 hours to prevent permanent account suspension. http://secure-banking-alert-update.com/login",
        "label": "spam",
        "notes": "Classic credential harvesting phishing email"
    },
    {
        "id": "spam_lottery_prize",
        "subject": "CONGRATULATIONS: You won $2,500,000 cash in the International Sweepstakes!",
        "body": "Official notification from the Global Lottery Bureau. Your email address was selected as the first prize winner of $2,500,000 cash! To claim your jackpot prize, reply urgently with your full name, phone number, and mailing address. No purchase required. Claim today!",
        "label": "spam",
        "notes": "Classic prize/lottery sweepstakes scam"
    },
    {
        "id": "spam_crypto_investment",
        "subject": "Guaranteed 500% profit with automated Bitcoin AI trading bot",
        "body": "Double your crypto in 7 days! Our proprietary AI trading algorithm guarantees risk-free returns. Minimum deposit only $250. Thousands of investors are earning $5,000 weekly passive income. Sign up right now to get a $100 free bonus deposit!",
        "label": "spam",
        "notes": "High-yield investment program (HYIP) / crypto scam"
    },
    {
        "id": "spam_pharmacy_discount",
        "subject": "Exclusive 80% discount on generic medications - No prescription needed",
        "body": "Order authentic meds online at unbeatable discount prices. Fast discreet express shipping right to your door. Buy Viagra, Cialis, antibiotics and pain relief without prescription. Click here to claim your limited-time 80% coupon!",
        "label": "spam",
        "notes": "Online counterfeit pharmacy spam"
    },
    {
        "id": "spam_wire_invoice",
        "subject": "OVERDUE INVOICE #89211 - Final Notice Before Legal Action",
        "body": "Please find attached the past due invoice #89211 for services rendered last month. Amount due: $8,420.00. Payment must be wired immediately to the escrow account provided below to avoid immediate referral to collection agencies and legal action.",
        "label": "spam",
        "notes": "Business email compromise (BEC) / fake invoice scam"
    },
    {
        "id": "ham_sprint_meeting",
        "subject": "Sprint planning agenda and Q4 deliverables review",
        "body": "Hi team, Just a reminder that our sprint planning session is scheduled for tomorrow at 10:00 AM in Conference Room B. Please review the updated backlog and Jira tickets beforehand. We need to finalize the engineering roadmap and API endpoint specifications for Q4. Thanks, Alex",
        "label": "ham",
        "notes": "Standard internal company team collaboration email"
    },
    {
        "id": "ham_code_review",
        "subject": "Pull Request #412: Implement user authentication middleware and unit tests",
        "body": "Hi Sarah, I've pushed the code review changes for PR #412 addressing the session token expiration. All unit tests and integration tests are passing in CI. Could you take another look at the middleware handler when you have a moment? The diff is ready for review.",
        "label": "ham",
        "notes": "Developer engineering review notification"
    },
    {
        "id": "ham_lunch_coworker",
        "subject": "Lunch today around 12:30?",
        "body": "Hey, a few of us are walking over to the taco truck down the street around 12:30 PM. Let me know if you want to join or if you need me to grab something for you!",
        "label": "ham",
        "notes": "Casual peer-to-peer personal communication"
    },
    {
        "id": "ham_client_contract",
        "subject": "Signed consulting agreement and project kickoff timeline",
        "body": "Dear David, Thank you for sending over the revised consulting agreement. We have signed the document and attached the executed PDF. We are excited to begin work on the data pipeline next Monday. Let's schedule our introductory kickoff call for Tuesday afternoon.",
        "label": "ham",
        "notes": "Formal business client correspondence"
    },
    {
        "id": "ham_doctor_appointment",
        "subject": "Confirmation: Medical appointment with Dr. Martinez on Nov 18",
        "body": "This is a confirmation that your annual wellness visit with Dr. Martinez is scheduled for Wednesday, November 18 at 2:15 PM at Valley Health Clinic. Please arrive 15 minutes early and bring your photo ID and insurance card. If you need to reschedule, call our office at 555-0192.",
        "label": "ham",
        "notes": "Healthcare confirmation notice"
    },
    {
        "id": "borderline_free_conference",
        "subject": "Claim your complimentary ticket to the Annual Developer Tech Summit",
        "body": "Join thousands of software engineers and machine learning researchers at the Global Tech Summit next month. Attendance is free for early registrations. Keynote talks on artificial intelligence, distributed systems, and open source development. Reserve your free seat now before spots fill up!",
        "label": "ham",
        "notes": "Subtle/borderline: contains 'free', 'claim', 'now', but is a legitimate tech event announcement"
    },
    {
        "id": "borderline_urgent_deadline",
        "subject": "Urgent: Project deadline moved up to Friday 5 PM",
        "body": "Team, executive leadership requested that we submit the final project presentation and financial forecast by Friday 5 PM instead of next week. Please review your assigned slides and send me your sections before Thursday noon so we can consolidate. Thank you for your hard work.",
        "label": "ham",
        "notes": "Subtle/borderline: contains 'Urgent', but is a legitimate workplace deadline email"
    }
]


def load_sms_dataset(filepath: str = "data/sms_dataset.tsv") -> Tuple[List[str], List[str]]:
    """Load the SMS spam collection dataset (tab-separated: label \\t text)."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")

    texts: List[str] = []
    labels: List[str] = []

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 2:
                label = row[0].strip().lower()
                text = row[1].strip()
                if label in ["ham", "spam"] and text:
                    labels.append(label)
                    texts.append(text)

    return texts, labels


def create_curated_email_dataset(filepath: str = "data/emails.csv") -> Tuple[List[str], List[str]]:
    """Write and load the curated email dataset into CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Expanded pool of email records to ensure rich vocabulary and solid class balance
    raw_records = list(CURATED_BENCHMARK_EMAILS)

    # Additional diverse email samples
    extra_emails = [
        # Ham
        ("ham", "Quarterly Budget Forecast Meeting", "Hi Brenda, Please see the attached spreadsheet with the Q3 expenditure figures. Let's discuss discrepancies in the marketing budget during our 1-on-1."),
        ("ham", "Flight Confirmation - Booking Reference #XK891", "Your flight from Boston to Chicago is confirmed for departure on December 4th. Terminal 2, Gate B14. Check in online 24 hours prior to flight."),
        ("ham", "Feedback on draft proposal", "Mark, I reviewed the draft you sent over. Overall looks very solid. I left a few minor comments on pages 3 and 7 regarding server sizing."),
        ("ham", "Gym membership renewal receipt", "Thank you for your recent payment of $45.00 for your monthly gym membership. Your receipt is attached for your records."),
        ("ham", "Dinner plans this weekend?", "Hey mom, just checking in to see if you and dad want to get dinner together on Saturday. Let me know what time works best."),
        ("ham", "University Research Symposium Submission Accepted", "Dear Author, We are pleased to inform you that your paper 'Bayesian Methods in Information Retrieval' has been accepted for oral presentation at the upcoming symposium."),
        ("ham", "Security alert: New login from your Chrome browser on Windows", "We noticed a new sign-in to your work account on Windows. If this was you, no action is needed. If not, contact your IT department."),
        ("ham", "Invoice for web design services - Project Milestone 2", "Attached please find invoice INV-2026-088 for the frontend UI development completed last week. Net 30 payment terms apply."),
        ("ham", "Updated office holiday schedule", "All employees: Please note that the main office will be closed on Thanksgiving Thursday and Friday. Normal business operations resume Monday."),
        ("ham", "Code repository migrated to new organization", "Team, the Git repository migration is complete. Please update your git remote URL to point to the new org repo before pushing any new branches."),
        
        # Spam
        ("spam", "ACT NOW: Claim your $500 Amazon Gift Card reward", "Congratulations! You have been randomly selected from our customer database to receive an exclusive $500 Amazon gift certificate. Click here to confirm your claim before midnight!"),
        ("spam", "URGENT SECURITY NOTICE: Compromised Bank Account", "Your online banking profile was flagged for suspicious activity. Immediate identity re-verification is required. Follow this link to avoid total account lockdown."),
        ("spam", "Pre-approved for $50,000 personal loan with 0% interest", "Bad credit? No problem! You are pre-approved for an instant cash loan up to $50,000. No collateral, no background check. Instant wire transfer to your bank. Apply today!"),
        ("spam", "Burn belly fat fast with this miracle secret extract", "Doctors are stunned by this brand new organic dietary formula. Lose 15 lbs in 10 days without exercise or dieting. Buy 2 bottles and get 3 free! Order today while supplies last."),
        ("spam", "Dearest in Christ, Transfer of $18.5 Million Funds", "I am Mrs. Fatima Al-Mansoor, widow to the late diplomat. I have deposited $18.5M in a security company abroad. I need an honest person to transfer these funds for charitable causes. 30% share is yours."),
        ("spam", "Cheap designer replica watches and luxury handbags", "Rolex, Gucci, Prada at 90% discount! 100% waterproof Swiss movement replicas. Worldwide free delivery. Shop the exclusive clearance warehouse catalog now!"),
        ("spam", "Work from home making $800 daily stuffing envelopes", "Earn huge income working only 2 hours a day from the comfort of your couch. No skills required. Immediate payout every Friday. Register your email now to get the startup kit!"),
        ("spam", "WINNER NOTIFICATION: Apple iPhone 16 Pro Giveaway", "You are the runner-up winner of the annual smartphone prize drawing! To claim your brand new iPhone, pay only $1 for shipping and confirm your delivery postal address."),
        ("spam", "Refinance your mortgage at historic lowest rate ever", "Homeowners alert! Government backed mortgage relief program allows you to drop your monthly payments by half. Check eligibility in 60 seconds with no cost or obligation."),
        ("spam", "Stop hair loss permanently with clinically proven serum", "Regrow thick, healthy hair in just 3 weeks! 100% money back guarantee. Limited stock available. Click here to claim your trial bottle!")
    ]

    for label, subj, body in extra_emails:
        raw_records.append({
            "id": f"email_{len(raw_records)}",
            "subject": subj,
            "body": body,
            "label": label,
            "notes": "Curated email"
        })

    texts: List[str] = []
    labels: List[str] = []

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label", "subject", "body", "full_text"])
        for r in raw_records:
            full_text = f"Subject: {r['subject']}\n\n{r['body']}"
            writer.writerow([r["id"], r["label"], r["subject"], r["body"], full_text])
            texts.append(full_text)
            labels.append(r["label"])

    return texts, labels


def train_test_split(
    texts: List[str],
    labels: List[str],
    test_size: float = 0.2,
    random_seed: int = 42,
    stratify: bool = True
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Split dataset into train and test sets with optional stratification."""
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length")

    rng = random.Random(random_seed)

    if not stratify:
        indices = list(range(len(texts)))
        rng.shuffle(indices)
        split_idx = int(len(texts) * (1.0 - test_size))
        train_idx = indices[:split_idx]
        test_idx = indices[split_idx:]
    else:
        # Stratify by class
        ham_indices = [i for i, lbl in enumerate(labels) if lbl == 'ham']
        spam_indices = [i for i, lbl in enumerate(labels) if lbl == 'spam']

        rng.shuffle(ham_indices)
        rng.shuffle(spam_indices)

        ham_split = int(len(ham_indices) * (1.0 - test_size))
        spam_split = int(len(spam_indices) * (1.0 - test_size))

        train_idx = ham_indices[:ham_split] + spam_indices[:spam_split]
        test_idx = ham_indices[ham_split:] + spam_indices[spam_split:]

        rng.shuffle(train_idx)
        rng.shuffle(test_idx)

    train_texts = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    test_texts = [texts[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    return train_texts, train_labels, test_texts, test_labels
