let SessionLoad = 1
if &cp | set nocp | endif
let s:so_save = &g:so | let s:siso_save = &g:siso | setg so=0 siso=0 | setl so=-1 siso=-1
let v:this_session=expand("<sfile>:p")
silent only
silent tabonly
cd ~\OneDrive\Documents\Code\GitHub\fractals\julia\Fractals
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
let s:shortmess_save = &shortmess
if &shortmess =~ 'A'
  set shortmess=aoOA
else
  set shortmess=aoO
endif
badd +70 \WINDOWS\SystemApps\MicrosoftWindows.Client.CBS_cw5n1h2txyewy\src\matrixfractal.jl
badd +156 \Users\emmet\Documents\Code\GitHub\fractals\rectangular_generators\ifs_data\fractint.ifs
badd +26 \Users\emmet\Documents\Code\GitHub\fractals\rectangular_generators\ifs_data\Default.ifs
badd +1 \Users\emmet\Documents\Code\GitHub\fractals\rectangular_generators\ifs_data\Edgar.ifs
badd +222 \WINDOWS\SystemApps\MicrosoftWindows.Client.CBS_cw5n1h2txyewy\src\matrixfractal_refactored.jl
badd +1 \WINDOWS\SystemApps\MicrosoftWindows.Client.CBS_cw5n1h2txyewy\src\matrixfractal_faster.jl
badd +403 src\matrixfractal_faster.jl
badd +53 src\fractal_1.jl
badd +0 :q
argglobal
%argdel
$argadd \WINDOWS\SystemApps\MicrosoftWindows.Client.CBS_cw5n1h2txyewy\src\matrixfractal.jl
edit src\matrixfractal_faster.jl
let s:save_splitbelow = &splitbelow
let s:save_splitright = &splitright
set splitbelow splitright
wincmd _ | wincmd |
vsplit
1wincmd h
wincmd w
wincmd _ | wincmd |
split
1wincmd k
wincmd w
let &splitbelow = s:save_splitbelow
let &splitright = s:save_splitright
wincmd t
let s:save_winminheight = &winminheight
let s:save_winminwidth = &winminwidth
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
exe 'vert 1resize ' . ((&columns * 78 + 78) / 157)
exe '2resize ' . ((&lines * 19 + 20) / 40)
exe 'vert 2resize ' . ((&columns * 78 + 78) / 157)
exe '3resize ' . ((&lines * 18 + 20) / 40)
exe 'vert 3resize ' . ((&columns * 78 + 78) / 157)
argglobal
balt src\fractal_1.jl
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let &fdl = &fdl
let s:l = 404 - ((19 * winheight(0) + 19) / 38)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 404
normal! 0
wincmd w
argglobal
if bufexists(fnamemodify("src\fractal_1.jl", ":p")) | buffer src\fractal_1.jl | else | edit src\fractal_1.jl | endif
balt src\matrixfractal_faster.jl
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let &fdl = &fdl
let s:l = 53 - ((6 * winheight(0) + 9) / 19)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 53
normal! 0
wincmd w
argglobal
terminal ++curwin ++cols=78 ++rows=18 ++type=winpty julia
let s:term_buf_9 = bufnr()
balt \WINDOWS\SystemApps\MicrosoftWindows.Client.CBS_cw5n1h2txyewy\src\matrixfractal_faster.jl
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 8985 - ((17 * winheight(0) + 9) / 18)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 8985
normal! 07|
wincmd w
exe 'vert 1resize ' . ((&columns * 78 + 78) / 157)
exe '2resize ' . ((&lines * 19 + 20) / 40)
exe 'vert 2resize ' . ((&columns * 78 + 78) / 157)
exe '3resize ' . ((&lines * 18 + 20) / 40)
exe 'vert 3resize ' . ((&columns * 78 + 78) / 157)
tabnext 1
if exists('s:wipebuf') && len(win_findbuf(s:wipebuf)) == 0
  silent exe 'bwipe ' . s:wipebuf
endif
unlet! s:wipebuf
set winheight=1 winwidth=20
let &shortmess = s:shortmess_save
let &winminheight = s:save_winminheight
let &winminwidth = s:save_winminwidth
let s:sx = expand("<sfile>:p:r")."x.vim"
if filereadable(s:sx)
  exe "source " . fnameescape(s:sx)
endif
let &g:so = s:so_save | let &g:siso = s:siso_save
doautoall SessionLoadPost
unlet SessionLoad
" vim: set ft=vim :
