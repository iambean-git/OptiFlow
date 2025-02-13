import { Navigate, Outlet, useLocation } from "react-router-dom";
import { useRecoilValue } from "recoil";

import { loginToken, userRole } from "../recoil/LoginAtom";

export default function ProtectedRoute({ requiredRole }) {
    const token = useRecoilValue(loginToken);
    const role = useRecoilValue(userRole);
    const location = useLocation();

    if (!token) {
        return <Navigate to="/unauthorized" replace state={{reason : "not_logged_in"}}/>;
    }

    // Admin 페이지 접근 제한
    if (requiredRole && role !== requiredRole) {
        return <Navigate to="/unauthorized" replace state={{ from: location, reason:"not_admin" }} />;
    }

    return <Outlet />;
}
