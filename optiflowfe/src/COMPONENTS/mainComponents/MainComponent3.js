import "./MainComponent.css";

export default function MainComponent3() {
  return (
    <div className="inner">
      <video
        src="/videos/수돗물.mp4"
        loop
        muted
        autoPlay
        className="background-video" // 백그라운드 비디오 클래스 이름 변경
      />
      <img src="/images/logo_square.png" className="size-96" alt="logo" />
      <section className="text-center">
        <p>
          {" "}
          어둠 속에서 빛나는 효율, 스마트 솔루션이 에너지 패러다임을 바꿉니다.
        </p>
        <p></p>
      </section>
    </div>
  );
}
