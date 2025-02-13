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

    if (reason === "not_admin") {
      alert("관리자 권한이 필요합니다.");
      navigate("/");
    } else {
      if (token) {
        console.log("토큰:", token);
        navigate("/notfound");
        return;
      }
      alert("로그인이 필요합니다.");
      navigate("/login");
    }
  }, []);

  return (
    <div>

    </div>
  )
}
