import { useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { useRecoilValue } from "recoil";

import { loginToken } from "../recoil/LoginAtom";

export default function NotLogined() {
  const token = useRecoilValue(loginToken);
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    const reason = location.state?.reason;
console.log(reason);
    if (reason === "not_admin") {
      setTimeout(() => {
        alert("관리자 권한이 필요합니다.");
        navigate("/");
      }, 0);
      return;
    }

    if (token) {
      console.log("토큰:", token);
      setTimeout(() => {
        navigate("/notfound");
      }, 0);
      return;
    }

    setTimeout(() => {
      alert("로그인이 필요합니다.");
      navigate("/login");
    }, 0);
  }, []);

  return <div></div>;
}
