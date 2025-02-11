import {useState} from 'react'
import AnimatedInput from '../components/ui/AnimatedInput';
export default function FreeTest() {
  const [name, setName] = useState("");
  const [contact, setContact] = useState("");
  const [email, setEmail] = useState("");
  const [location, setLocation] = useState("");
  const [question, setQuestion] = useState("");

  return (
    <div className='w-full h-full flex flex-col items-center'>
      <header className='w-full py-5 flex justify-center border-b'>
        <img src='/images/logo_text_blue.png' alt="logo"
          className='h-14' />
      </header>
      <main className='w-10/12 bg-red-50 flex flex-col justify-center items-center pt-10'>
        <AnimatedInput
          type={"text"} label={"이름"} value={name}
          onChange={(e) => setName(e.target.value)}
        />
        <AnimatedInput
          type={"text"} label={"연락처"} value={contact}
          onChange={(e) => setContact(e.target.value)}
        />
        <AnimatedInput
          type={"text"} label={"메일"} value={email}
          onChange={(e) => setEmail(e.target.value)}
        />
        <AnimatedInput
          type={"text"} label={"이용 희망 지역"} value={location}
          onChange={(e) => setLocation(e.target.value)}
          detail={"서비스를 이용하실 지역을 입력해주세요."}
        />
        <AnimatedInput
          type={"text"} label={"이용 희망 지역"} value={location}
          onChange={(e) => setLocation(e.target.value)}
          detail={"서비스를 이용하실 지역을 입력해주세요."}
        />
        <AnimatedInput
          type={"text"} label={"기타 문의사항"} value={question}
          onChange={(e) => setQuestion(e.target.value)}
          textarea={true}
        />
      </main>

    </div>
  )
}
