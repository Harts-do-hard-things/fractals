let SessionLoad = 1
if &cp | set nocp | endif
let s:so_save = &g:so | let s:siso_save = &g:siso | setg so=0 siso=0 | setl so=-1 siso=-1
let v:this_session=expand("<sfile>:p")
silent only
silent tabonly
cd ~\Documents\Code\GitHub\fractals\julia\Fractals
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
let s:shortmess_save = &shortmess
if &shortmess =~ 'A'
  set shortmess=aoOA
else
  set shortmess=aoO
endif
badd +194 src\matrixfractal.jl
badd +156 \Users\emmet\Documents\Code\GitHub\fractals\rectangular_generators\ifs_data\fractint.ifs
badd +26 \Users\emmet\Documents\Code\GitHub\fractals\rectangular_generators\ifs_data\Default.ifs
badd +99 \Users\emmet\Documents\Code\GitHub\fractals\rectangular_generators\ifs_data\Edgar.ifs
badd +7 src\fractal_1.jl
badd +12 src\matrixfractal_refactored.jl
argglobal
%argdel
$argadd src\matrixfractal.jl
edit src\matrixfractal.jl
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
exe '2resize ' . ((&lines * 21 + 20) / 40)
exe 'vert 2resize ' . ((&columns * 78 + 78) / 157)
exe '3resize ' . ((&lines * 16 + 20) / 40)
exe 'vert 3resize ' . ((&columns * 78 + 78) / 157)
argglobal
balt src\matrixfractal_refactored.jl
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
let s:l = 111 - ((1 * winheight(0) + 19) / 38)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 111
normal! 030|
wincmd w
argglobal
if bufexists(fnamemodify("src\matrixfractal_refactored.jl", ":p")) | buffer src\matrixfractal_refactored.jl | else | edit src\matrixfractal_refactored.jl | endif
balt src\matrixfractal.jl
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
let s:l = 216 - ((4 * winheight(0) + 10) / 21)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 216
normal! 021|
wincmd w
argglobal
terminal ++curwin ++cols=78 ++rows=16 ++type=winpty julia
let s:term_buf_13 = bufnr()
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 177 - ((14 * winheight(0) + 8) / 16)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 177
normal! 05|
wincmd w
exe 'vert 1resize ' . ((&columns * 78 + 78) / 157)
exe '2resize ' . ((&lines * 21 + 20) / 40)
exe 'vert 2resize ' . ((&columns * 78 + 78) / 157)
exe '3resize ' . ((&lines * 16 + 20) / 40)
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
