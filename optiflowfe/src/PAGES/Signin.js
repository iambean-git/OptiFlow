import '../CSS/signin.css';
import loginBG from "../ASSETS/images/loginBG.png";
export default function Signin() {

  //로그인 처리
  const handleClick = () => {
    console.log("click");
  }

  return (
    <div className='w-full h-screen bg-red-50 flex justify-center items-center' style={{backgroundImage:`url(${loginBG})`}}>

      <div className='w-[550px] h-[420px] bg-white bg-opacity-60 rounded-md flex flex-col items-center justify-center'>
        <div className="w-[178px] h-[37px] mb-8">
          <img src="/images/logo_text.png" alt='logo'></img>
        </div>

        <label htmlFor="email" className="input_label">
          ID</label>
        <input id="email" type="text" placeholder="username"
          className="input_box mb-[30px]"></input>

        <label htmlFor="password" className="input_label">
          Password</label>
        <input id="password" type="password" placeholder="********"
          className="input_box mb-[30px]"></input>

        <button className='w-[420px] h-[40px] mt-4 bg-[#1D5673] text-white rounded-md font-Pretendard'
                onClick={handleClick}> LOGIN </button>
      </div>
    </div>
  )
}
