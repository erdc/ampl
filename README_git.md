## Steps for working with git
1. [GitLab setup](#GitLab-setup)


# GitLab setup
This section provides instructions on setting up git and is required to befor a user will be able to donload the AMPL source code. Check if Git is already installed by the command `git`.  If it is not installed, go to [Git](https://git-scm.com/downloads) and download it based on your system.  Git will be used later.

Next, you will need to clone the AMPL Git repo into your AMPL code directory `<ampl_code_dir>` to access the AMPL codes. In order to access the repo from your system you will need an access token or ssh key.

There are two methods to access Git: HTTPS and SSH key
HTTPS prompts you with userid and access token information everytime you push/pull, with SSH you don't have to remember or enter any access token. Choose either the HTTPS or SSH key method.  The recommended method is the SSH key.


## Option 1: Get SSH key (recommended)
First we need to create a ssh key.  If you already have a ssh key, make sure that it is added to Gitlab. Instructions on adding the ssh key to git can be found in section "Create the ssh key" below

### Create the ssh key
This is needed to clone the repo, as well as pull/push from the repo.

https://docs.gitlab.com/ee/user/ssh.html#add-an-ssh-key-to-your-gitlab-account

1. Open a terminal. Note fow Windows: Please open anaconda power shell as an administrator for the commands to work.
2. Run ssh-keygen -t followed by the key type and an optional comment. This comment is included in the .pub file that’s created. You may want to use an email address for the comment.

```shell
ssh-keygen -o -t rsa -b 4096 -C "<email@myEmail.com>"
```

3. Press Enter. Output similar to the following is displayed:

`Generating public/private rsa key pair.`

Enter file in which to save the key `(/home/user/.ssh/id_rsa)`

4. Accept the suggested filename and directory


### Add SSH Key to GitLab
1. To use SSH with GitLab, copy your public key to your GitLab account: You can do this manually, copy the contents of id_rsa.pub

```shell
# Linux/OSX based machines
vi /p/home/<username>>/.ssh/id_rsa.pub

# Windows
C:\Users\<myUserName>\.ssh
```

2. Add the new SSH key in Git lab using the steps below:\
Sign in to GitLab.\
On the left sidebar, select your avatar.\
Select Edit profile.\
On the left sidebar, select SSH Keys.\
Select Add new key.\
\
In the Key box, paste the contents of your public key. If you manually copied the key, make sure you copy the entire key, which starts with ssh-rsa, ssh-dss, ecdsa-sha2-nistp256, ecdsa-sha2-nistp384, ecdsa-sha2-nistp521, ssh-ed25519, sk-ecdsa-sha2-nistp256@openssh.com, or sk-ssh-ed25519@openssh.com, and may end with a comment.\
\
In the Title box, type a description, like Work Laptop or Home Workstation.\
Optional. Select the Usage type of the key. Authentication & Signing.\
Optional. Remove the expiration date
Select Add key.

For ease of reference the following is the windows command to copy the git public key. This can also be found in the git document linked in this section:

```shell
cat ~/.ssh/id_rsa.pub | clip
```

## Option 2: Get HTTPS Access Token (needed to clone the repo, as well as pull/push)

1. Access the AMPL Git repo from your browser.
2. Once in the repo, navigate to `Settings Access Tokens` on the left-hand side of the screen.
3. To create a new token, click `Add new token` on the ‘Project Access Tokens’ page.
4. Give your token a name, an expiration date, select a role if asked, if provided the option check all of the boxes under `Select scopes`, and then select `Create project access token`.
5. Be sure to save your access token somewhere once it is generated because this will be the only time it will be given to you.
