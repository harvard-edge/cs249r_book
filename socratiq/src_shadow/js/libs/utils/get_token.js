export async function get_token() {
    const fakeToken = "fake_token_12345";
    localStorage.setItem("token_avaya", fakeToken);
    return fakeToken;
}
