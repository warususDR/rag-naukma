import { GoogleLogin } from "@react-oauth/google";
import { useNavigate } from "react-router";
import { useAuth } from "../components/AuthProvider";
import { useEffect } from "react";
import { toast } from "sonner";
import logoUrl from "../assets/logo.svg";

function decodeJwt(token: string): Record<string, string> {
  const [, payload] = token.split(".");
  return JSON.parse(atob(payload.replace(/-/g, "+").replace(/_/g, "/")));
}

export default function Login() {
  const { credential, login } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    if (credential) navigate("/", { replace: true });
  }, [credential, navigate]);

  return (
    <div className="min-h-screen bg-naukma-light flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-lg p-8 w-full max-w-sm text-center">
        <div className="w-20 h-20 rounded-full overflow-hidden bg-white ring-4 ring-naukma-navy/10 mx-auto mb-6">
          <img src={logoUrl} alt="НаУКМА" className="w-full h-full object-cover" />
        </div>
        <h1 className="text-2xl font-bold text-naukma-navy mb-1">
          <span className="text-naukma-gold">НаУКМА</span> RAG
        </h1>
        <p className="text-sm text-gray-500 mb-8">
          Запитай про Могилянку
        </p>
        <div className="flex justify-center">
          <GoogleLogin
            onSuccess={(res) => {
              const token = res.credential!;
              const claims = decodeJwt(token);
              const email = claims["email"] ?? "";
              const name = claims["name"] ?? "";
              const picture = claims["picture"] ?? "";
              login(token, email, name, picture);
              navigate("/", { replace: true });
            }}
            onError={() => toast.error("Помилка входу. Спробуйте ще раз.")}
            text="signin_with"
          />
        </div>
        <p className="text-xs text-gray-400 mt-4">Увійдіть через Google акаунт</p>
      </div>
    </div>
  );
}
