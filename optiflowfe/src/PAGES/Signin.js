import "../css/signin.css";
import "../css/videoStyle.css"; 

import loginBG from "../assets/images/loginBG.png";
import { useRef, useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";

import { useSetRecoilState } from "recoil";
import { loginToken, userName } from "../recoil/LoginAtom";

export default function Signin() {
  const navigate = useNavigate();
  const [username, setUsername] = useState("");
  const [pwd, setPwd] = useState("");

  const setToken = useSetRecoilState(loginToken);
  const setUser = useSetRecoilState(userName);

  const [btnDisabled, setBtnDisabled] = useState(true);

  const usernameRef = useRef();
  const passwordRef = useRef();

  useEffect(() => {
    // console.log("username : ", username);
    // console.log("pwd : ", pwd);
    if (username.length > 0 && pwd.length > 0) setBtnDisabled(false);
    else setBtnDisabled(true);
  }, [username, pwd]);

  //로그인 처리
  const handleClick = () => {
    console.log("click");
    fetchLogoin();
  };

  // 엔터(Enter) 키 입력 시 로그인 실행
  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !btnDisabled) {
      handleClick();
    }
  };

  const fetchLogoin = async () => {
    const url = "http://10.125.121.226:8080/login";

    const loginData = {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        username: usernameRef.current.value,
        password: passwordRef.current.value,
      }),
    };

    try {
      const resp = await fetch(url, loginData);

      if (!resp.ok) {
        throw new Error(`[LOGIN] HTTP error! status: ${resp.status}`);
      }
      // 응답 헤더에서 Authorization 값을 추출
      const token = resp.headers.get("Authorization");
      const userID = resp.headers.get("username");
      // console.log("authHeader : ", authHeader);
      // console.log("로그인된 아이디 :",userID);
      if (token) {
        sessionStorage.setItem("token", token);
        sessionStorage.setItem("username", userID);

        // **Recoil 상태 즉시 업데이트**
        setToken(token);
        setUser(userID);

        // console.log("login 성공");
        navigate("/dashboard");
      } else {
        console.log("Authorization header not found");
        alert("로그인에 실패하였습니다. ID 또는 Password를 다시 확인해주세요.");
        window.location.reload();
      }
    } catch (err) {
      console.error("Error fetching data:", err);
      alert("로그인에 실패하였습니다. ID 또는 Password를 다시 확인해주세요.");
      window.location.reload();
    }
  };

  return (

    <div className="main-component-container">
      <video
        src="/videos/waterdrop1.mp4"
        loop
        muted
        autoPlay
        className="background-video" // 백그라운드 비디오 클래스 이름 변경
      />

      <div className="content-overlay">
        <div className="w-[380px] h-[420px] bg-white bg-opacity-60 rounded-md flex flex-col items-center justify-center">
          <div className="w-[178px] h-[37px] mb-8">
            <img src="/images/logo_text.png" alt="logo"></img>
          </div>

          <label htmlFor="email" className="input_label">
            ID
          </label>
          <input
            id="email"
            type="text"
            placeholder="username"
            ref={usernameRef}
            onChange={(e) => setUsername(e.target.value)}
            className="input_box mb-[30px]"
          ></input>

          <label htmlFor="password" className="input_label">
            Password
          </label>
          <input
            id="password"
            type="password"
            placeholder="********"
            ref={passwordRef}
            onChange={(e) => setPwd(e.target.value)}
            onKeyDown={handleKeyDown}  // 🔹 엔터 키 입력 감지
            className="input_box mb-[30px]"
          ></input>

          <button
            className="w-[80%] h-[40px] mt-4 bg-[#2C7CB8] text-white rounded-md duration-500
                          disabled:cursor-not-allowed  disabled:bg-[#76aad1] disabled:text-[#FFFFFF70] disabled:opacity-90"
            onClick={handleClick}
            disabled={btnDisabled}
          >
            {" "}
            LOGIN{" "}
          </button>
        </div>
      </div>
    </div>


  );
}
