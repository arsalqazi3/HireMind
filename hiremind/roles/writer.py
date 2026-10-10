"""Writer: cover letter for the chosen job, then a claim check."""
from hiremind.schemas import Job, Profile


def write(profile: Profile, job: Job) -> str:
    raise NotImplementedError


def unsupported_claims(letter: str, profile: Profile) -> list[str]:
    """Return skills, experience, education or projects in the letter that are not in the profile."""
    raise NotImplementedError


def write_checked(profile: Profile, job: Job, max_tries: int = 2) -> str:
    """Write the letter, then remove unsupported claims or write it again."""
    raise NotImplementedError
