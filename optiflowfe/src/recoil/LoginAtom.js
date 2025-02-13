import { atom, selector } from "recoil";

export const loginToken = atom({
    key: "logoinToken",
    default : sessionStorage.getItem("token") ? sessionStorage.getItem("token") : null
});

export const userName = atom({
    key: "userName",
    default : sessionStorage.getItem("username") ? sessionStorage.getItem("username") : null
});

export const userRole = atom({
    key: "userRole",
    default : sessionStorage.getItem("userRole") ? sessionStorage.getItem("userRole") : null
});