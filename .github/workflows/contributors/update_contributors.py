import json
import os

import numpy as np
import pandas as pd
import github
import requests
from absl import app
from absl import logging

CONTRIBUTORS_FILE = ".all-contributorsrc"

EXCLUDED_USERS = {"web-flow", "github-actions[bot]", "mrdragonbear", "jveejay"}

OWNER = "harvard-edge"
REPO = "cs249r_book"
BRANCH = "main"


def get_github_user_full_name(username):
  g = github.Github(os.environ["GITHUB_TOKEN"])
  try:
    user = g.get_user(username)
    return user.name
  except github.GithubException:
    return None


def get_github_user_email_address(username):
  g = github.Github(os.environ["GITHUB_TOKEN"])
  try:
    user = g.get_user(username)
    return user.email
  except github.GithubException:
    return None


def get_username_from_email(email):
  g = github.Github(os.environ["GITHUB_TOKEN"])
  try:
    user = g.get_user(email)
    return user.login, email
  except github.GithubException:
    return None, email


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
              {'user_full_name': user_full_name,
               'email_address': email_address})
      except ValueError as e:
        logging.error(
            f"Error parsing co-author: {line}. Co-author should be of the form: "
            f"'Co-authored-by: <name> <email>'. "
            f"Remember to include the angle brackets around the email."
        )
    return pd.DataFrame(co_author_data)


def main(_):
  token = os.environ["GITHUB_TOKEN"]
  headers = {"Authorization": f"token {token}"}
  data = []
  next_page = (
      f"https://api.github.com/repos/{OWNER}/{REPO}/commits?sha={BRANCH}&per_page=100"
  )
  last_page = None
  while next_page != last_page:
    print(f"Fetching page: {next_page}")
    res = requests.get(next_page, headers=headers)
    data.extend(res.json())
    next_page = res.links.get("next", {}).get("url", None)
    last_page = res.links.get("last", {}).get("url", None)

  commit_data = []
  for node in data:
    commit_message = node.get("commit", {}).get("message", None)
    commit_info = node.get("commit", None)
    commit_author_info = commit_info.get("author", None)
    commit_commiter_info = commit_info.get("committer", None)
    author_info = node.get("author", None)
    committer_info = node.get("committer", None)
    committer_login_info = (
        committer_info.get("login", None) if committer_info else None
    )
    user_full_name = None
    username = None

    if commit_author_info:
      user_full_name = commit_author_info["name"]
    elif commit_commiter_info:
      user_full_name = commit_commiter_info["name"]

    if author_info:
      username = author_info["login"]
    elif committer_login_info:
      username = committer_login_info["login"]

    commit_data.append(
        {
            "commit_message": commit_message,
            "user_full_name": user_full_name,
            "username": username,
        }
    )
  commit_data_df = pd.DataFrame(commit_data)
  users_from_api = commit_data_df["username"].unique().tolist()
  print("Users pulled from API: ", users_from_api)

  co_authors_list = [get_co_authors_from_commit_message(row["commit_message"])
                     for index, row in commit_data_df.iterrows()]
  co_authors_df = pd.concat(co_authors_list, ignore_index=True)

  commit_data_df.drop(columns=["commit_message"], inplace=True)
  commit_data_df = commit_data_df.merge(
      co_authors_df, how="left", on="user_full_name")
  commit_data_df.drop_duplicates(
      subset=["user_full_name", "username", "email_address"], inplace=True)

  # Try to get email addresses from GitHub API
  commit_data_df = commit_data_df.assign(
      email_address=commit_data_df.apply(
          lambda row: get_github_user_email_address(row['username'])
          if pd.isna(row['email_address']) and not pd.isna(row['username'])
          else row['email_address'],
          axis=1
      )
  )

  # Remove rows with excluded users
  commit_data_df = commit_data_df[~commit_data_df["username"].isin(
      EXCLUDED_USERS)]
  commit_data_df = commit_data_df[~commit_data_df["user_full_name"].isin(
      EXCLUDED_USERS)]
  commit_data_df = commit_data_df.fillna(value=np.nan)
  commit_data_df = commit_data_df.assign(
      name_length=commit_data_df['user_full_name'].str.len())
  commit_data_df = commit_data_df.sort_values(by='name_length', ascending=False)

  # Group by 'username' and aggregate
  aggregated_df = commit_data_df.groupby('username', as_index=False).first()
  aggregated_df = aggregated_df[["username", "user_full_name", "email_address"]]

  # Now as a last ditch effort, try to get the full name from the GitHub API
  # Only do this if user_full_name is the same as username or if user_full_name is NaN
  aggregated_df = aggregated_df.assign(
      user_full_name=aggregated_df.apply(
          lambda row: get_github_user_full_name(row['username'])
          if row['user_full_name'] == row['username'] or pd.isna(
              row['user_full_name'])
          else row['user_full_name'],
          axis=1
      )
  )

  # At this point, if we don't have a user_full_name, just use the username
  aggregated_df = aggregated_df.assign(
      user_full_name=aggregated_df.apply(
          lambda row: row['username']
          if pd.isna(row['user_full_name'])
          else row['user_full_name'],
          axis=1
      )
  )

  final_result = dict(
      projectName=REPO,
      projectOwner=OWNER,
      files=["contributors.qmd", "README.md"],
      contributors=[
          dict(
              login=row.username,
              name=row.user_full_name,
              avatar_url=f"https://avatars.githubusercontent.com/{row.username}",
              profile=f"https://github.com/{row.username}",
              contributions=[],
          )
          for row in aggregated_df.itertuples()
      ],
      repoType="github",
      contributorsPerLine=5,
      repoHost="https=//github.com",
      commitConvention="angular",
      skipCi=True,
  )

  json_string = json.dumps(
      final_result, indent=4
  )
  print(json_string)

  with open(CONTRIBUTORS_FILE, "w") as contrib_file:
    contrib_file.write(json_string)


if __name__ == "__main__":
  app.run(main)
