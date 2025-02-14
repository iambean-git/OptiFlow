import NavBar from "../components/NavBar";
import { useEffect, useState } from "react";
import { userName } from "../recoil/LoginAtom";
import { useRecoilValue } from "recoil";

export default function MyPage() {
  const labelCss = "w-full h-6 mb-2";
  const inputCss = "w-full px-4 py-3 flex items-center border rounded-lg mb-[30px] focus:outline-none focus:ring-2";

  const [password, setPwd] = useState(null);
  const [newPw, setNewPw] = useState(null);
  const [newPwCheck, setPwCheck] = useState(null);
  const [checkSame, setCheckSame] = useState(false);

  const handlePasswordChange = () => {
    console.log("비밀번호 변경 클릭");
  }

  useEffect(() => {
    if (!newPw) return;
    if (newPw === newPwCheck) setCheckSame(true);
    else setCheckSame(false);
  }, [newPw, newPwCheck]);

  return (
    <div className="w-full min-w-[1000px] h-screen bg-[#f2f2f2] ">
      <NavBar />
      <div className="w-full h-screen pl-[260px] flex flex-col justify-center items-center">
        <div className="w-1/4 px-10 py-8 bg-white rounded-md flex flex-col items-center justify-center">
          <div className="w-full h-1/4 flex justify-center items-center text-2xl mb-10 text-[#3b82f6] font-semibold ">
            비밀번호 변경
          </div>

          <label htmlFor="password" className={labelCss}>
            기존 비밀번호
          </label>
          <input
            id="password"
            type="password"
            placeholder="********"
            onChange={(e) => setPwd(e.target.value)}
            className={`${inputCss} focus:ring-[#3b82f6]`}
          ></input>

          <label htmlFor="new_pw" className={labelCss}>
            신규 비밀번호
          </label>
          <input
            id="new_pw"
            type="password"
            placeholder="********"
            onChange={(e) => setNewPw(e.target.value)}
            className={`${inputCss} focus:ring-[#3b82f6]`}
          ></input>

          <label htmlFor="new_pw_check" className={labelCss}>
            신규 비밀번호 확인
          </label>
          <input
            id="new_pw_check"
            type="password"
            placeholder="********"
            onChange={(e) => setPwCheck(e.target.value)}
            className={`${inputCss} ${newPwCheck && !checkSame ? "ring-2 ring-red-500" : "focus:ring-[#3b82f6]"}`}
          ></input>
          <span className={`${newPwCheck && !checkSame ? "block" : "hidden"} w-full ml-1.5 -mt-6 mb-4 text-xs text-red-500`} >
            신규 비밀번호가 일치하지 않습니다.
          </span>

          <button
            className="w-full py-2 my-4 bg-[#3b82f6] text-white rounded-md duration-500 
                          disabled:cursor-not-allowed disabled:opacity-45"
            disabled={!password || !newPw || !newPwCheck || !checkSame}
            onClick={handlePasswordChange}
          >
            변경하기
          </button>
        </div>
      </div>

    </div>
  )
}
