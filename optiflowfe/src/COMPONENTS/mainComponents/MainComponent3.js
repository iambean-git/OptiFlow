import "../../css/videoStyle.css"; 

export default function MainComponent3() {
  return (
    <div className="inner">

      <div className="w-2/5 h-full relative overflow-hidden"> {/* 부모 div에 relative 적용 */}
        <video
          // src="/videos/수돗물.mp4"
          src="/videos/waterdrop2.mp4"
          loop
          muted
          autoPlay
          className="absolute top-0 left-0 w-full h-full object-cover scale-x-[-1] z-[-1] scale-y-[1.2] object-[50%]"
        />
      </div>


      <div className="w-3/5 h-full">

      </div>

      {/* <section className="text-center">
        <p>
          {" "}
          어둠 속에서 빛나는 효율, 스마트 솔루션이 에너지 패러다임을 바꿉니다.
        </p>
        <p></p>
      </section> */}
    </div>
  );
}
