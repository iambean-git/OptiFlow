import NavBar from "../components/NavBar";
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { userName } from "../recoil/LoginAtom";
import { useRecoilValue } from "recoil";

export default function MyPage() {
  const server = process.env.REACT_APP_SERVER_ADDR;
  const navigate = useNavigate();
  const labelCss = "w-full h-6 mb-2";
  const inputCss =
    "w-full px-4 py-3 flex items-center border rounded-lg mb-[30px] focus:outline-none focus:ring-2";

  const [password, setPwd] = useState("");
  const [newPw, setNewPw] = useState("");
  const [newPwCheck, setPwCheck] = useState("");
  const [checkSame, setCheckSame] = useState(false);
  const username = useRecoilValue(userName);

  const handlePasswordChange = (e) => {
    e.preventDefault(); // 폼 기본 제출 방지
    console.log("비밀번호 변경 클릭", { password, newPw, newPwCheck });
    fetchPutNewPW();
  };

  const fetchPutNewPW = async () => {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 2000); // 2초 후 요청 중단
    const requestBody = {
      username: username,
      password: password,
      newpw: newPw
    };

    try {
      const url = `${server}/api/members/password`;
      const response = await fetch(url, {
        signal: controller.signal,
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody)
      });
      // console.log("💌[MyPage] 비밀번호 수정 response: ", response);
      clearTimeout(timeoutId); // 응답이 오면 타이머 제거
      if (!response.ok) {
        //기존 비밀번호 잘못 입력했을때 처리하기!!!! 
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      // ✅ 비밀번호 변경 후 state를 추가하여 대시보드로 이동
      navigate("/dashboard", { state: { passwordChanged: true }, replace: true });

    } catch (err) {
      console.error("❌[MyPage] 비밀번호 수정 실패:", err);
    } finally {
      // setLoading(false);
    }
  };

  useEffect(() => {
    if (!newPw) return;
    setCheckSame(newPw === newPwCheck);
  }, [newPw, newPwCheck]);

  return (
    <div className="w-full min-w-[1000px] h-screen bg-[#f2f2f2]">
      <NavBar />
      <div className="w-full h-screen pl-[260px] flex flex-col justify-center items-center">
        <div className="w-1/3 2xl:w-1/4 px-10 py-8 bg-white rounded-md flex flex-col items-center justify-center">
          <div className="w-full h-1/4 flex justify-center items-center text-2xl mb-10 text-[#3b82f6] font-semibold">
            비밀번호 변경
          </div>

          {/* 폼 추가 */}
          <form className="w-full flex flex-col" onSubmit={handlePasswordChange}>

            <input
              type="text"
              name="username"
              autoComplete="username"
              className="hidden"
            />

            <label htmlFor="password" className={labelCss}>
              기존 비밀번호
            </label>
            <input
              id="password"
              type="password"
              placeholder="********"
              onChange={(e) => setPwd(e.target.value)}
              autoComplete="current-password"
              className={`${inputCss} focus:ring-[#3b82f6]`}
            />

            <label htmlFor="new_pw" className={labelCss}>
              신규 비밀번호
            </label>
            <input
              id="new_pw"
              type="password"
              placeholder="********"
              onChange={(e) => setNewPw(e.target.value)}
              autoComplete="new-password"
              className={`${inputCss} focus:ring-[#3b82f6]`}
            />

            <label htmlFor="new_pw_check" className={labelCss}>
              신규 비밀번호 확인
            </label>
            <input
              id="new_pw_check"
              type="password"
              placeholder="********"
              onChange={(e) => setPwCheck(e.target.value)}
              autoComplete="new-password"
              className={`${inputCss} ${newPwCheck && !checkSame ? "ring-2 ring-red-500" : "focus:ring-[#3b82f6]"
                }`}
            />
            <span
              className={`${newPwCheck && !checkSame ? "block" : "hidden"
                } w-full ml-1.5 -mt-6 mb-4 text-xs text-red-500`}
            >
              신규 비밀번호가 일치하지 않습니다.
            </span>

            <button
              type="submit"
              className="w-full py-2 my-4 bg-[#3b82f6] text-white rounded-md duration-500 
                          disabled:cursor-not-allowed disabled:opacity-45"
              disabled={!password || !newPw || !newPwCheck || !checkSame}
            >
              변경하기
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
