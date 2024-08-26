import os
import json
import random
import hashlib

import requests
import pandas as pd
from absl import app
from absl import logging

CONTRIBUTORS_FILE = ".all-contributorsrc"

EXCLUDED_USERS = {
    "web-flow",
    "github-actions[bot]",
    "mrdragonbear",
    "jveejay",
    "Matthew Steward",
}

OWNER = "harvard-edge"
REPO = "cs249r_book"
BRANCH = "dev"
RESULTS_PER_PAGE = 1000


def get_user_data_from_username(username):
    headers = {"Authorization": f"token {os.environ['GITHUB_TOKEN']}"}
    res = requests.get(f"https://api.github.com/users/{username}", headers=headers)
    user_full_name = pd.NA
    email_address = pd.NA
    if res.status_code == 200:
        user_data = res.json()
        user_full_name = user_data["name"]
        email_address = user_data["email"]
    else:
        logging.error(f"Could not find user with username: {username}")
    return {
        "username": username,
        "user_full_name": user_full_name,
        "email_address": email_address,
    }


def get_user_data_from_email(email_address):
    headers = {"Authorization": f"token {os.environ['GITHUB_TOKEN']}"}
    res = requests.get(
        f"https://api.github.com/search/users?q={email_address}", headers=headers
    )
    username = pd.NA
    if res.status_code == 200:
        user_data = res.json()
        if user_data["total_count"] > 0:
            username = user_data["items"][0]["login"]
    else:
        logging.error(f"Could not find user with email address: {email_address}")
    return {
        "username": username,
        "user_full_name": pd.NA,
        "email_address": email_address,
    }


def get_co_authors_from_commit_message(commit_message):
    co_author_data = []
    if commit_message:
        lines = commit_message.splitlines()
        for line in lines:
            try:
                if line.startswith("Co-authored-by:"):
                    co_author = line.split(":")[1].strip()
                    user_full_name, email_address = co_author.split("<")
                    user_full_name = user_full_name.strip()
                    email_address = email_address.strip(">")
                    co_author_data.append(
                        {
                            "user_full_name": user_full_name,
                            "email_address": email_address,
                            "username": pd.NA,
                        }
                    )
            except ValueError as e:
                logging.error(
                    f"Error parsing co-author: {line}. Co-author should be of the form: "
                    f"'Co-authored-by: NAME <email>'. "
                    f"Remember to include the angle brackets around the email."
                )
        return pd.DataFrame(co_author_data)


def merge_user_full_names(row, col1, col2):
    """
    Merges two columns containing user full names based on the following criteria:
    - Takes the longest name that is not null and not an email address.

    Parameters:
    - row: A single row from the DataFrame.
    - col1: The first column name containing user full names.
    - col2: The second column name containing user full names.

    Returns:
    - The merged user full name based on the criteria.
    """

    def is_email(string):
        return isinstance(string, str) and "@" in string

    name1 = row[col1]
    name2 = row[col2]

    if (
        pd.notna(name1)
        and not is_email(name1)
        and (pd.isna(name2) or len(name1) >= len(name2))
    ):
        return name1
    elif pd.notna(name2) and not is_email(name2):
        return name2
    else:
        return pd.NA


def merge_email_addresses(row, col1, col2):
    """
    Merges two columns containing email addresses based on the following criteria:
    - Returns the email address that is not null.
    - If both email addresses are not null, it prioritizes the one that does not contain 'noreply.github.com'.

    Parameters:
    - row: A single row from the DataFrame.
    - col1: The first column name containing email addresses.
    - col2: The second column name containing email addresses.

    Returns:
    - The selected email address based on the criteria, or pd.NA if both are null.
    """

    email1 = row[col1]
    email2 = row[col2]

    # Check if either email is not null
    if pd.notna(email1) and "noreply.github.com" not in email1:
        return email1
    elif pd.notna(email2) and "noreply.github.com" not in email2:
        return email2
    elif pd.notna(email1):
        return email1
    elif pd.notna(email2):
        return email2
    else:
        return pd.NA


def main(_):
    token = os.environ["GITHUB_TOKEN"]
    headers = {"Authorization": f"token {token}"}
    data = []
    next_page = f"https://api.github.com/repos/{OWNER}/{REPO}/commits?sha={BRANCH}&per_page={RESULTS_PER_PAGE}"
    last_page = None
    while next_page != last_page:
        print(f"Fetching page: {next_page}")
        res = requests.get(next_page, headers=headers)
        data.extend(res.json())
        next_page = res.links.get("next", {}).get("url", None)
        last_page = res.links.get("last", {}).get("url", None)

    # Parse the commit response data
    commit_data = []
    for node in data:
        commit_message = node.get("commit", {}).get("message", pd.NA)
        commit_info = node.get("commit", None)
        commit_author_info = commit_info.get("author", None)
        commit_commiter_info = commit_info.get("committer", None)
        author_info = node.get("author", None)
        committer_info = node.get("committer", None)
        committer_login_info = (
            committer_info.get("login", None) if committer_info else None
        )
        user_full_name = pd.NA
        user_login = pd.NA
        user_email_address = pd.NA

        if commit_author_info:
            user_full_name = commit_author_info["name"]
            user_email_address = commit_author_info["email"]
        elif commit_commiter_info:
            user_full_name = commit_commiter_info["name"]

        if author_info:
            user_login = author_info["login"]
        elif committer_login_info:
            user_login = committer_login_info["login"]

        commit_data.append(
            {
                "commit_message": commit_message,
                "user_full_name": user_full_name,
                "email_address": user_email_address,
                "username": user_login,
            }
        )
    commit_data_df = pd.DataFrame(commit_data)

    # Parse the co-author data from the commit messages
    co_authors_list = [
        get_co_authors_from_commit_message(row["commit_message"])
        for index, row in commit_data_df.iterrows()
    ]
    co_authors_df = pd.concat(co_authors_list, ignore_index=True)

    # All co-authors must have an email address, so look up info and replace
    # with whatever is on GitHub
    for index, row in co_authors_df.iterrows():
        user_data = get_user_data_from_email(row.email_address)
        co_authors_df.loc[index, "username"] = user_data["username"]

    # Remove excluded users
    co_authors_df = co_authors_df[
        ~co_authors_df["username"].isin(EXCLUDED_USERS)
        & ~co_authors_df["user_full_name"].isin(EXCLUDED_USERS)
    ]
    commit_data_df = commit_data_df[
        ~commit_data_df["username"].isin(EXCLUDED_USERS)
        & ~commit_data_df["user_full_name"].isin(EXCLUDED_USERS)
    ]

    # Count contributions in each DataFrame
    co_authors_df["co_author_count"] = co_authors_df.groupby("email_address")[
        "email_address"
    ].transform("count")

    # Create a combined key using username and email_address to handle
    # cases with missing usernames. Users can commit without specifying
    # their username, but they should have an email address.
    commit_data_df["user_key"] = commit_data_df["username"].combine_first(
        commit_data_df["email_address"]
    )

    # Count the number of commits per user (grouped by user_key)
    commit_data_df["commit_count"] = commit_data_df.groupby("user_key")[
        "user_key"
    ].transform("count")

    # Drop the user_key if it's no longer needed
    commit_data_df.drop(columns=["user_key"], inplace=True)

    # Since we have the count, remove duplicates
    commit_data_df = commit_data_df.drop(columns=["commit_message"])
    co_authors_df.drop_duplicates(inplace=True)
    commit_data_df.drop_duplicates(inplace=True)

    # Now try to find all users with GitHub API
    for index, row in commit_data_df.iterrows():
        if not pd.isna(row["username"]):
            user_data = get_user_data_from_username(row["username"])
            if not pd.isna(user_data["username"]):
                commit_data_df.loc[index, "user_full_name"] = user_data[
                    "user_full_name"
                ]
            if not pd.isna(user_data["email_address"]):
                commit_data_df.loc[index, "email_address"] = user_data["email_address"]
        elif not pd.isna(row["email_address"]):
            user_data = get_user_data_from_email(row["email_address"])
            if not pd.isna(user_data["username"]):
                commit_data_df.loc[index, "username"] = user_data["username"]
            if not pd.isna(user_data["user_full_name"]):
                commit_data_df.loc[index, "user_full_name"] = user_data[
                    "user_full_name"
                ]
        else:
            logging.error(
                "Could not find user data for commit: " f"{row['commit_message']}"
            )

    co_authors_with_username = co_authors_df[~co_authors_df["username"].isna()]
    co_authors_without_username = co_authors_df[co_authors_df["username"].isna()]

    # First merge: on username
    merged_df = co_authors_with_username.merge(
        commit_data_df,
        how="outer",
        on=["username"],
        suffixes=("_co", "_commit"),
        indicator=True,
    )

    # Calculate total contributions after first merge
    merged_df["total_contributions"] = merged_df["co_author_count"].fillna(
        0
    ) + merged_df["commit_count"].fillna(0)

    # Merge user full name columns
    merged_df["user_full_name"] = merged_df["user_full_name"] = merged_df.apply(
        merge_user_full_names,
        col1="user_full_name_commit",
        col2="user_full_name_co",
        axis=1,
    )

    # Merge email address columns
    merged_df["email_address"] = merged_df.apply(
        merge_email_addresses,
        col1="email_address_commit",
        col2="email_address_co",
        axis=1,
    )

    # Drop unnecessary columns
    merged_df = merged_df.drop(
        columns=[
            "_merge",
            "co_author_count",
            "commit_count",
            "user_full_name_co",
            "user_full_name_commit",
            "email_address_co",
            "email_address_commit",
        ]
    )
    merged_df.drop_duplicates(inplace=True)

    # Second merge: co-authors without username on email
    merged_df = co_authors_without_username.merge(
        merged_df,
        how="outer",
        on="email_address",
        suffixes=("_co_no_user", ""),
        indicator=True,
    )

    # Update total contributions after second merge
    merged_df["total_contributions"] = merged_df["total_contributions"].fillna(
        0
    ) + merged_df["co_author_count"].fillna(0)

    # Merge user full name columns
    merged_df["user_full_name"] = merged_df["user_full_name"] = merged_df.apply(
        merge_user_full_names,
        col1="user_full_name",
        col2="user_full_name_co_no_user",
        axis=1,
    )

    # Remove unnecessary columns
    merged_df = merged_df.drop(
        columns=["_merge", "co_author_count", "username_co_no_user"]
    )

    # Merge the user full name columns
    merged_df["user_full_name"] = merged_df.apply(
        merge_user_full_names,
        col1="user_full_name",
        col2="user_full_name_co_no_user",
        axis=1,
    )
    merged_df = merged_df.drop(columns=["user_full_name_co_no_user"])

    # Get name length to figure out which full name to use
    merged_df = merged_df.assign(name_length=merged_df["user_full_name"].str.len())
    merged_df = merged_df.fillna(pd.NA)
    merged_df = merged_df.sort_values(
        by=["total_contributions", "name_length"], ascending=False
    )

    # Separate rows with and without usernames
    df_with_username = merged_df.dropna(subset=["username"])
    df_without_username = merged_df[merged_df["username"].isna()]

    # Group by username, and take the user_full_name with the most characters for rows with usernames
    df_with_username = df_with_username.groupby("username", as_index=False).first()

    # Remove rows from df_without_username where the user_full_name matches a user_full_name in df_with_username.
    # We do this to avoid duplicate entries for the same user. Without a
    # username, we do not know if two rows are the same user.
    df_without_username = df_without_username[
        ~df_without_username["user_full_name"].isin(df_with_username["user_full_name"])
    ]

    # Combine the grouped rows with usernames and the original rows without usernames
    merged_df = pd.concat([df_with_username, df_without_username], ignore_index=True)

    def generate_gravatar_url(name):
        random.seed(name)
        name_list = list(name)
        random.shuffle(name_list)
        name = "".join(name_list)
        name_hash = hashlib.md5(name.encode("utf-8")).hexdigest()
        return f"https://www.gravatar.com/avatar/{name_hash}?d=identicon&s=100"

    # Update avatar_url
    merged_df["avatar_url"] = merged_df.apply(
        lambda row: (
            generate_gravatar_url(row["user_full_name"])
            if pd.isna(row["username"])
            else f"https://avatars.githubusercontent.com/{row['username']}"
        ),
        axis=1,
    )

    # Update profile URL
    merged_df["profile"] = merged_df.apply(
        lambda row: (
            "https://github.com/harvard-edge/cs249r_book/graphs/contributors"
            if pd.isna(row["username"])
            else f"https://github.com/{row['username']}"
        ),
        axis=1,
    )

    # Sort by number of commits
    merged_df.sort_values(by="total_contributions", ascending=False, inplace=True)

    final_result = dict(
        projectName=REPO,
        projectOwner=OWNER,
        files=["contents/contributors.qmd", "README.md"],
        contributors=[
            dict(
                login=(
                    row.username if not pd.isna(row.username) else row.user_full_name
                ),
                name=(
                    row.user_full_name
                    if not pd.isna(row.user_full_name)
                    else row.username
                ),
                avatar_url=row.avatar_url,
                profile=row.profile,
                contributions=[],
            )
            for row in merged_df.itertuples()
        ],
        repoType="github",
        contributorsPerLine=5,
        repoHost="https://github.com",
        commitConvention="angular",
        skipCi=True,
    )

    json_string = json.dumps(final_result, indent=4)
    print(json_string)

    with open(CONTRIBUTORS_FILE, "w") as contrib_file:
        contrib_file.write(json_string)


if __name__ == "__main__":
    app.run(main)
